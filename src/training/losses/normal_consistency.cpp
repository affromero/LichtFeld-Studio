/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "normal_consistency.hpp"
#include "lfs/kernels/normal_consistency.cuh"
#include <format>

namespace lfs::training::losses {

    void NormalConsistencyLoss::ensure_buffers(
        size_t n_gaussians,
        size_t height,
        size_t width,
        size_t k) {

        // Only reallocate if dimensions changed
        if (allocated_n_ != n_gaussians ||
            allocated_h_ != height ||
            allocated_w_ != width ||
            allocated_k_ != k) {

            gaussian_normals_ = lfs::core::Tensor::empty(
                {n_gaussians, 3}, lfs::core::Device::CUDA);

            depth_normals_ = lfs::core::Tensor::empty(
                {height, width, 3}, lfs::core::Device::CUDA);

            loss_per_pixel_ = lfs::core::Tensor::empty(
                {height, width}, lfs::core::Device::CUDA);

            size_t num_pixels = height * width;
            size_t num_blocks = std::min((num_pixels + 255) / 256, size_t(1024));
            reduction_buffer_ = lfs::core::Tensor::empty(
                {num_blocks}, lfs::core::Device::CUDA);

            loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);

            allocated_n_ = n_gaussians;
            allocated_h_ = height;
            allocated_w_ = width;
            allocated_k_ = k;
        }
    }

    std::expected<std::pair<lfs::core::Tensor, NormalConsistencyLoss::Context>, std::string>
    NormalConsistencyLoss::forward(
        const lfs::core::Tensor& quaternions,
        const lfs::core::Tensor& scales,
        const lfs::core::Tensor& depth_map,
        const lfs::core::Tensor& blend_weights,
        const lfs::core::Tensor& gaussian_indices,
        float camera_fx,
        float camera_fy,
        const Params& params,
        lfs::core::Tensor& quaternions_grad,
        lfs::core::Tensor& scales_grad) {

        try {
            // Early return for zero weight
            if (params.weight <= 0.0f) {
                auto zero_loss = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                Context ctx{
                    .loss_tensor = zero_loss,
                    .gaussian_normals = lfs::core::Tensor(),
                    .depth_normals = lfs::core::Tensor()};
                return std::make_pair(zero_loss, ctx);
            }

            // Validate inputs
            if (quaternions.device() != lfs::core::Device::CUDA) {
                return std::unexpected("quaternions must be on CUDA device");
            }
            if (scales.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scales must be on CUDA device");
            }
            if (depth_map.device() != lfs::core::Device::CUDA) {
                return std::unexpected("depth_map must be on CUDA device");
            }
            if (blend_weights.device() != lfs::core::Device::CUDA) {
                return std::unexpected("blend_weights must be on CUDA device");
            }
            if (gaussian_indices.device() != lfs::core::Device::CUDA) {
                return std::unexpected("gaussian_indices must be on CUDA device");
            }

            // Validate shapes
            if (quaternions.shape().rank() != 2 || quaternions.shape()[1] != 4) {
                return std::unexpected("quaternions must have shape [N, 4]");
            }
            if (scales.shape().rank() != 2 || scales.shape()[1] != 3) {
                return std::unexpected("scales must have shape [N, 3]");
            }
            if (quaternions.shape()[0] != scales.shape()[0]) {
                return std::unexpected("quaternions and scales must have same N");
            }
            if (depth_map.shape().rank() != 2) {
                return std::unexpected("depth_map must have shape [H, W]");
            }
            if (blend_weights.shape().rank() != 3) {
                return std::unexpected("blend_weights must have shape [H, W, K]");
            }
            if (gaussian_indices.shape().rank() != 3) {
                return std::unexpected("gaussian_indices must have shape [H, W, K]");
            }

            size_t N = quaternions.shape()[0];
            size_t H = depth_map.shape()[0];
            size_t W = depth_map.shape()[1];
            size_t K = blend_weights.shape()[2];

            if (blend_weights.shape()[0] != H || blend_weights.shape()[1] != W) {
                return std::unexpected("blend_weights shape must match depth_map [H, W, K]");
            }
            if (gaussian_indices.shape()[0] != H || gaussian_indices.shape()[1] != W ||
                gaussian_indices.shape()[2] != K) {
                return std::unexpected("gaussian_indices shape must match blend_weights");
            }

            // Ensure workspace buffers are allocated
            ensure_buffers(N, H, W, K);

            // Zero the loss scalar
            loss_scalar_.zero_();

            // Convert scales from log space to linear space for normal computation
            auto scales_linear = scales.exp();

            // Step 1: Compute Gaussian normals
            lfs::training::kernels::launch_compute_gaussian_normals(
                quaternions.ptr<float>(),
                scales_linear.ptr<float>(),
                gaussian_normals_.ptr<float>(),
                static_cast<int>(N),
                nullptr);

            // Step 2: Compute depth-derived surface normals
            lfs::training::kernels::launch_compute_depth_normals(
                depth_map.ptr<float>(),
                depth_normals_.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                camera_fx,
                camera_fy,
                nullptr);

            // Step 3: Compute fused forward + backward for efficiency
            lfs::training::kernels::launch_fused_normal_consistency(
                quaternions.ptr<float>(),
                scales.ptr<float>(),  // Pass log-space scales, kernel handles exp()
                depth_normals_.ptr<float>(),
                blend_weights.ptr<float>(),
                gaussian_indices.ptr<int>(),
                loss_scalar_.ptr<float>(),
                quaternions_grad.ptr<float>(),
                scales_grad.ptr<float>(),
                static_cast<int>(N),
                static_cast<int>(H),
                static_cast<int>(W),
                static_cast<int>(K),
                params.weight,
                nullptr);

            // Normalize loss by number of valid pixels (optional)
            // For now, we keep the raw sum for consistency with other losses

            Context ctx{
                .loss_tensor = loss_scalar_,
                .gaussian_normals = gaussian_normals_.clone(),
                .depth_normals = depth_normals_.clone()};

            return std::make_pair(loss_scalar_, ctx);

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Error in NormalConsistencyLoss::forward: {}", e.what()));
        }
    }

    // ==========================================================================
    // Helper function implementations
    // ==========================================================================

    lfs::core::Tensor compute_gaussian_normals(
        const lfs::core::Tensor& quaternions,
        const lfs::core::Tensor& scales) {

        if (!quaternions.is_valid() || !scales.is_valid()) {
            throw std::runtime_error("compute_gaussian_normals: invalid input tensors");
        }

        if (quaternions.device() != lfs::core::Device::CUDA) {
            throw std::runtime_error("compute_gaussian_normals: quaternions must be on CUDA");
        }
        if (scales.device() != lfs::core::Device::CUDA) {
            throw std::runtime_error("compute_gaussian_normals: scales must be on CUDA");
        }

        size_t N = quaternions.shape()[0];
        auto normals = lfs::core::Tensor::empty({N, 3}, lfs::core::Device::CUDA);

        lfs::training::kernels::launch_compute_gaussian_normals(
            quaternions.ptr<float>(),
            scales.ptr<float>(),
            normals.ptr<float>(),
            static_cast<int>(N),
            nullptr);

        return normals;
    }

    lfs::core::Tensor compute_depth_normals(
        const lfs::core::Tensor& depth_map,
        float fx,
        float fy) {

        if (!depth_map.is_valid()) {
            throw std::runtime_error("compute_depth_normals: invalid depth_map");
        }

        if (depth_map.device() != lfs::core::Device::CUDA) {
            throw std::runtime_error("compute_depth_normals: depth_map must be on CUDA");
        }

        if (depth_map.shape().rank() != 2) {
            throw std::runtime_error("compute_depth_normals: depth_map must be 2D [H, W]");
        }

        size_t H = depth_map.shape()[0];
        size_t W = depth_map.shape()[1];

        auto normals = lfs::core::Tensor::empty({H, W, 3}, lfs::core::Device::CUDA);

        lfs::training::kernels::launch_compute_depth_normals(
            depth_map.ptr<float>(),
            normals.ptr<float>(),
            static_cast<int>(H),
            static_cast<int>(W),
            fx,
            fy,
            nullptr);

        return normals;
    }

} // namespace lfs::training::losses
