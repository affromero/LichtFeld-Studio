/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "multiview_regularization.hpp"
#include "lfs/kernels/multiview_regularization.cuh"
#include <format>

namespace lfs::training::losses {

    // =============================================================================
    // Helper: Convert Camera to CameraParams for GPU
    // =============================================================================

    static kernels::CameraParams camera_to_params(const lfs::core::Camera& cam) {
        kernels::CameraParams params;

        // Intrinsics
        auto [fx, fy, cx, cy] = cam.get_intrinsics();
        params.fx = fx;
        params.fy = fy;
        params.cx = cx;
        params.cy = cy;

        // Extrinsics: R and T
        // R is [3, 3], T is [3]
        auto R_cpu = cam.R().cpu();
        auto T_cpu = cam.T().cpu();

        // Copy rotation matrix (row-major)
        for (int i = 0; i < 9; ++i) {
            params.R[i] = R_cpu.ptr<float>()[i];
        }

        // Copy translation
        for (int i = 0; i < 3; ++i) {
            params.t[i] = T_cpu.ptr<float>()[i];
        }

        params.width = cam.image_width();
        params.height = cam.image_height();

        return params;
    }

    // =============================================================================
    // Workspace management
    // =============================================================================

    void MultiViewRegularization::ensure_workspace(int height, int width) {
        if (allocated_height_ == height && allocated_width_ == width) {
            return;
        }

        // Allocate buffers
        warped_coords_ = lfs::core::Tensor::empty({static_cast<size_t>(height),
                                                    static_cast<size_t>(width), 2},
                                                   lfs::core::Device::CUDA);
        cycle_coords_ = lfs::core::Tensor::empty({static_cast<size_t>(height),
                                                   static_cast<size_t>(width), 2},
                                                  lfs::core::Device::CUDA);
        reprojection_error_ = lfs::core::Tensor::empty({static_cast<size_t>(height),
                                                         static_cast<size_t>(width)},
                                                        lfs::core::Device::CUDA);
        confidence_weights_ = lfs::core::Tensor::empty({static_cast<size_t>(height),
                                                         static_cast<size_t>(width)},
                                                        lfs::core::Device::CUDA);
        ncc_scores_ = lfs::core::Tensor::empty({static_cast<size_t>(height),
                                                 static_cast<size_t>(width)},
                                                lfs::core::Device::CUDA);
        loss_buffer_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);

        // Temporary buffer for reductions
        size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
        size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
        temp_reduction_ = lfs::core::Tensor::empty({num_blocks * 2}, lfs::core::Device::CUDA);

        allocated_height_ = height;
        allocated_width_ = width;
    }

    // =============================================================================
    // Input validation
    // =============================================================================

    std::expected<void, std::string> MultiViewRegularization::validate_inputs(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam,
        const lfs::core::Tensor& ref_depth,
        const lfs::core::Tensor& ref_normal,
        const lfs::core::Tensor& ref_image,
        const lfs::core::Tensor& neighbor_image) {

        // Check devices
        if (ref_depth.device() != lfs::core::Device::CUDA) {
            return std::unexpected("ref_depth must be on CUDA device");
        }
        if (ref_normal.device() != lfs::core::Device::CUDA) {
            return std::unexpected("ref_normal must be on CUDA device");
        }
        if (ref_image.device() != lfs::core::Device::CUDA) {
            return std::unexpected("ref_image must be on CUDA device");
        }
        if (neighbor_image.device() != lfs::core::Device::CUDA) {
            return std::unexpected("neighbor_image must be on CUDA device");
        }

        // Check depth shape: [H, W]
        if (ref_depth.ndim() != 2) {
            return std::unexpected(
                std::format("ref_depth must be 2D [H, W], got {} dims", ref_depth.ndim()));
        }

        // Check normal shape: [H, W, 3]
        if (ref_normal.ndim() != 3 || ref_normal.shape()[2] != 3) {
            return std::unexpected(
                std::format("ref_normal must be [H, W, 3], got shape {}",
                            ref_normal.shape().str()));
        }

        // Check shapes match
        if (ref_depth.shape()[0] != ref_normal.shape()[0] ||
            ref_depth.shape()[1] != ref_normal.shape()[1]) {
            return std::unexpected("ref_depth and ref_normal shapes must match in H, W");
        }

        // Check image shapes
        if (ref_image.ndim() < 2 || ref_image.ndim() > 3) {
            return std::unexpected("ref_image must be [H, W] or [H, W, C]");
        }
        if (neighbor_image.ndim() < 2 || neighbor_image.ndim() > 3) {
            return std::unexpected("neighbor_image must be [H, W] or [H, W, C]");
        }

        // Check image dimensions match depth
        if (ref_image.shape()[0] != ref_depth.shape()[0] ||
            ref_image.shape()[1] != ref_depth.shape()[1]) {
            return std::unexpected("ref_image dimensions must match ref_depth");
        }

        return {};
    }

    // =============================================================================
    // Relative pose computation
    // =============================================================================

    std::pair<lfs::core::Tensor, lfs::core::Tensor> compute_relative_pose(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam) {

        // Get rotation matrices and translations
        auto R_ref = ref_cam.R();      // [3, 3]
        auto T_ref = ref_cam.T();      // [3]
        auto R_neighbor = neighbor_cam.R();  // [3, 3]
        auto T_neighbor = neighbor_cam.T();  // [3]

        // R_rel = R_neighbor @ R_ref.T
        auto R_ref_t = R_ref.transpose();
        auto R_rel = R_neighbor.mm(R_ref_t);

        // t_rel = T_neighbor - R_rel @ T_ref
        auto t_rel = T_neighbor - R_rel.mm(T_ref.reshape({3, 1})).reshape({3});

        return {R_rel, t_rel};
    }

    lfs::core::Tensor compute_depth_homography(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam,
        float depth,
        const lfs::core::Tensor& normal) {

        // Get camera parameters
        auto [fx_r, fy_r, cx_r, cy_r] = ref_cam.get_intrinsics();
        auto [fx_n, fy_n, cx_n, cy_n] = neighbor_cam.get_intrinsics();

        // Build intrinsics matrices
        auto K_ref = lfs::core::Tensor::zeros({3, 3}, lfs::core::Device::CUDA);
        auto K_neighbor = lfs::core::Tensor::zeros({3, 3}, lfs::core::Device::CUDA);

        // K_ref
        std::vector<float> k_ref_data = {
            fx_r, 0.0f, cx_r,
            0.0f, fy_r, cy_r,
            0.0f, 0.0f, 1.0f
        };
        auto K_ref_cpu = lfs::core::Tensor::from_vector(k_ref_data, {3, 3}, lfs::core::Device::CPU);
        K_ref.copy_from(K_ref_cpu.cuda());

        // K_neighbor
        std::vector<float> k_neighbor_data = {
            fx_n, 0.0f, cx_n,
            0.0f, fy_n, cy_n,
            0.0f, 0.0f, 1.0f
        };
        auto K_neighbor_cpu = lfs::core::Tensor::from_vector(k_neighbor_data, {3, 3}, lfs::core::Device::CPU);
        K_neighbor.copy_from(K_neighbor_cpu.cuda());

        // Get relative pose
        auto [R_rel, t_rel] = compute_relative_pose(ref_cam, neighbor_cam);

        // Compute K_ref_inv
        // For a standard intrinsic matrix, the inverse is:
        // [1/fx,  0,  -cx/fx]
        // [0,  1/fy,  -cy/fy]
        // [0,    0,      1  ]
        std::vector<float> k_ref_inv_data = {
            1.0f / fx_r, 0.0f, -cx_r / fx_r,
            0.0f, 1.0f / fy_r, -cy_r / fy_r,
            0.0f, 0.0f, 1.0f
        };
        auto K_ref_inv = lfs::core::Tensor::from_vector(k_ref_inv_data, {3, 3}, lfs::core::Device::CUDA);

        // Compute H = K_neighbor @ (R - t @ n^T / d) @ K_ref_inv
        // where n is the normal and d is the depth

        // t @ n^T
        auto t_col = t_rel.reshape({3, 1});
        auto n_row = normal.reshape({1, 3});
        auto t_nT = t_col.mm(n_row);  // [3, 3]

        // R - t @ n^T / d
        auto mid = R_rel - t_nT.div(depth);

        // H = K_neighbor @ mid @ K_ref_inv
        auto H = K_neighbor.mm(mid).mm(K_ref_inv);

        return H;
    }

    // =============================================================================
    // Main loss computation
    // =============================================================================

    std::expected<lfs::core::Tensor, std::string> MultiViewRegularization::compute_loss(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam,
        const lfs::core::Tensor& ref_depth,
        const lfs::core::Tensor& ref_normal,
        const lfs::core::Tensor& ref_image,
        const lfs::core::Tensor& neighbor_image,
        const MultiViewRegConfig& config) {

        try {
            // Validate inputs
            auto validation = validate_inputs(ref_cam, neighbor_cam, ref_depth, ref_normal,
                                              ref_image, neighbor_image);
            if (!validation) {
                return std::unexpected(validation.error());
            }

            int height = static_cast<int>(ref_depth.shape()[0]);
            int width = static_cast<int>(ref_depth.shape()[1]);

            // Ensure workspace is allocated
            ensure_workspace(height, width);

            // Convert cameras to GPU-friendly format
            auto ref_params = camera_to_params(ref_cam);
            auto neighbor_params = camera_to_params(neighbor_cam);

            // Compute relative pose
            auto [R_rel, t_rel] = compute_relative_pose(ref_cam, neighbor_cam);
            auto R_rel_cpu = R_rel.cpu();
            auto t_rel_cpu = t_rel.cpu();

            // Compute inverse relative pose for back-warping
            auto R_rel_inv = R_rel.transpose();
            auto t_rel_inv = (R_rel_inv.mm(t_rel.reshape({3, 1})) * -1.0f).reshape({3});
            auto R_rel_inv_cpu = R_rel_inv.cpu();
            auto t_rel_inv_cpu = t_rel_inv.cpu();

            // Get image dimensions
            int neighbor_height = static_cast<int>(neighbor_image.shape()[0]);
            int neighbor_width = static_cast<int>(neighbor_image.shape()[1]);
            int channels = (ref_image.ndim() == 3) ? static_cast<int>(ref_image.shape()[2]) : 1;

            // Separate warped coordinates into u and v
            auto warped_u_slice = warped_coords_.slice(2, 0, 1).squeeze(2);  // [H, W]
            auto warped_v_slice = warped_coords_.slice(2, 1, 2).squeeze(2);  // [H, W]

            // Step 1: Forward warp from reference to neighbor
            kernels::launch_depth_warp_forward(
                ref_depth.ptr<float>(),
                ref_normal.ptr<float>(),
                warped_u_slice.ptr<float>(),
                warped_v_slice.ptr<float>(),
                ref_params,
                neighbor_params,
                R_rel_cpu.ptr<float>(),
                t_rel_cpu.ptr<float>(),
                height,
                width,
                config.min_depth,
                nullptr);

            // Step 2: Inverse warp from neighbor back to reference
            // For simplicity, we assume neighbor has same depth map structure
            // In practice, you'd need to sample the neighbor's depth
            auto cycle_u_slice = cycle_coords_.slice(2, 0, 1).squeeze(2);  // [H, W]
            auto cycle_v_slice = cycle_coords_.slice(2, 1, 2).squeeze(2);  // [H, W]

            kernels::launch_depth_warp_inverse(
                warped_u_slice.ptr<float>(),
                warped_v_slice.ptr<float>(),
                ref_depth.ptr<float>(),  // Use ref depth as proxy for neighbor depth
                cycle_u_slice.ptr<float>(),
                cycle_v_slice.ptr<float>(),
                ref_params,
                neighbor_params,
                R_rel_inv_cpu.ptr<float>(),
                t_rel_inv_cpu.ptr<float>(),
                height,
                width,
                neighbor_height,
                neighbor_width,
                config.min_depth,
                nullptr);

            // Step 3: Compute cycle reprojection error and confidence weights
            kernels::launch_cycle_error_and_weights(
                cycle_u_slice.ptr<float>(),
                cycle_v_slice.ptr<float>(),
                reprojection_error_.ptr<float>(),
                confidence_weights_.ptr<float>(),
                height,
                width,
                config.error_threshold,
                nullptr);

            // Step 4: Compute NCC photometric matching (if weight > 0)
            if (config.photometric_weight > 0.0f) {
                kernels::launch_ncc_matching(
                    ref_image.ptr<float>(),
                    neighbor_image.ptr<float>(),
                    warped_u_slice.ptr<float>(),
                    warped_v_slice.ptr<float>(),
                    ncc_scores_.ptr<float>(),
                    height,
                    width,
                    neighbor_height,
                    neighbor_width,
                    channels,
                    config.patch_size,
                    nullptr);
            } else {
                ncc_scores_.fill_(1.0f);  // Perfect match if not using photometric
            }

            // Step 5: Combine losses with reduction
            kernels::launch_multiview_loss_reduction(
                reprojection_error_.ptr<float>(),
                ncc_scores_.ptr<float>(),
                confidence_weights_.ptr<float>(),
                loss_buffer_.ptr<float>(),
                temp_reduction_.ptr<float>(),
                height,
                width,
                config.photometric_weight,
                config.geometric_weight,
                nullptr);

            return loss_buffer_;

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Error in MultiViewRegularization::compute_loss: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> MultiViewRegularization::compute_loss_with_grad(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam,
        const lfs::core::Tensor& ref_depth,
        const lfs::core::Tensor& ref_normal,
        const lfs::core::Tensor& ref_image,
        const lfs::core::Tensor& neighbor_image,
        lfs::core::Tensor& ref_depth_grad,
        lfs::core::Tensor& ref_normal_grad,
        const MultiViewRegConfig& config) {

        try {
            // First compute forward pass
            auto loss_result = compute_loss(ref_cam, neighbor_cam, ref_depth, ref_normal,
                                            ref_image, neighbor_image, config);
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            // Validate gradient tensors
            if (ref_depth_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("ref_depth_grad must be on CUDA device");
            }
            if (ref_normal_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("ref_normal_grad must be on CUDA device");
            }
            if (ref_depth_grad.shape() != ref_depth.shape()) {
                return std::unexpected("ref_depth_grad shape must match ref_depth");
            }
            if (ref_normal_grad.shape() != ref_normal.shape()) {
                return std::unexpected("ref_normal_grad shape must match ref_normal");
            }

            int height = static_cast<int>(ref_depth.shape()[0]);
            int width = static_cast<int>(ref_depth.shape()[1]);

            // Convert cameras to GPU format
            auto ref_params = camera_to_params(ref_cam);
            auto neighbor_params = camera_to_params(neighbor_cam);

            // Compute relative pose
            auto [R_rel, t_rel] = compute_relative_pose(ref_cam, neighbor_cam);
            auto R_rel_cpu = R_rel.cpu();
            auto t_rel_cpu = t_rel.cpu();

            // Compute total weight for normalization
            float total_weight = confidence_weights_.sum_scalar();
            if (total_weight < 1e-6f) {
                // No valid correspondences, zero gradients
                return loss_result;
            }

            // Get cycle coordinates
            auto cycle_u_slice = cycle_coords_.slice(2, 0, 1).squeeze(2);
            auto cycle_v_slice = cycle_coords_.slice(2, 1, 2).squeeze(2);

            // Backward pass
            kernels::launch_multiview_backward(
                reprojection_error_.ptr<float>(),
                cycle_u_slice.ptr<float>(),
                cycle_v_slice.ptr<float>(),
                confidence_weights_.ptr<float>(),
                ref_depth.ptr<float>(),
                ref_normal.ptr<float>(),
                ref_depth_grad.ptr<float>(),
                ref_normal_grad.ptr<float>(),
                ref_params,
                neighbor_params,
                R_rel_cpu.ptr<float>(),
                t_rel_cpu.ptr<float>(),
                height,
                width,
                config.geometric_weight,
                total_weight,
                nullptr);

            return loss_result;

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Error in MultiViewRegularization::compute_loss_with_grad: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
