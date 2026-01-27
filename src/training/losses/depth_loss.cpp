/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_loss.hpp"
#include "lfs/kernels/depth_loss.cuh"
#include <format>

namespace lfs::training::losses {

    // ------------------------------
    // BUFFER MANAGEMENT
    // ------------------------------

    void DepthLoss::ensure_buffers(const std::vector<size_t>& shape, size_t num_blocks) {
        // Only reallocate if shape or num_blocks changed
        if (allocated_shape_ != shape || allocated_num_blocks_ != num_blocks) {
            lfs::core::TensorShape tshape(shape);
            grad_buffer_ = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            grad_x_buffer_ = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            grad_y_buffer_ = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);

            if (num_blocks > 0) {
                reduction_buffer_ = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            }

            allocated_shape_ = shape;
            allocated_num_blocks_ = num_blocks;
        }
    }

    // ------------------------------
    // L1 DEPTH LOSS
    // ------------------------------

    std::expected<lfs::core::Tensor, std::string> DepthLoss::compute_l1_loss(
        const lfs::core::Tensor& rendered_depth,
        const lfs::core::Tensor& gt_depth,
        const lfs::core::Tensor& valid_mask,
        lfs::core::Tensor& grad_out) {

        size_t H = rendered_depth.size(0);
        size_t W = rendered_depth.size(1);
        size_t N = H * W;
        size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

        // Ensure buffers are allocated
        ensure_buffers(rendered_depth.shape().dims(), num_blocks);

        // Launch fused L1 depth loss kernel
        lfs::training::kernels::launch_fused_depth_l1_loss(
            rendered_depth.ptr<float>(),
            gt_depth.ptr<float>(),
            valid_mask.ptr<float>(),
            grad_out.ptr<float>(),
            loss_scalar_.ptr<float>(),
            reduction_buffer_.ptr<float>(),
            H,
            W,
            nullptr);

        return loss_scalar_;
    }

    // ------------------------------
    // GRADIENT SMOOTHNESS LOSS
    // ------------------------------

    std::expected<lfs::core::Tensor, std::string> DepthLoss::compute_smoothness_loss(
        const lfs::core::Tensor& rendered_depth,
        const lfs::core::Tensor& valid_mask,
        lfs::core::Tensor& grad_out) {

        size_t H = rendered_depth.size(0);
        size_t W = rendered_depth.size(1);
        size_t N = H * W;
        size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

        // Ensure buffers are allocated
        ensure_buffers(rendered_depth.shape().dims(), num_blocks);

        // Zero out gradient output before accumulation
        grad_out.zero_();

        // Launch gradient smoothness kernel
        lfs::training::kernels::launch_depth_gradient_smoothness(
            rendered_depth.ptr<float>(),
            valid_mask.ptr<float>(),
            grad_out.ptr<float>(),
            loss_scalar_.ptr<float>(),
            reduction_buffer_.ptr<float>(),
            H,
            W,
            nullptr);

        return loss_scalar_;
    }

    // ------------------------------
    // EDGE-AWARE SMOOTHNESS LOSS
    // ------------------------------

    std::expected<lfs::core::Tensor, std::string> DepthLoss::compute_edge_aware_smoothness(
        const lfs::core::Tensor& rendered_depth,
        const lfs::core::Tensor& valid_mask,
        const lfs::core::Tensor& rendered_rgb,
        lfs::core::Tensor& grad_out) {

        size_t H = rendered_depth.size(0);
        size_t W = rendered_depth.size(1);
        size_t N = H * W;
        size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

        // Ensure buffers are allocated
        ensure_buffers(rendered_depth.shape().dims(), num_blocks);

        // Zero out gradient output before accumulation
        grad_out.zero_();

        // Launch edge-aware gradient smoothness kernel
        lfs::training::kernels::launch_depth_edge_aware_smoothness(
            rendered_depth.ptr<float>(),
            valid_mask.ptr<float>(),
            rendered_rgb.ptr<float>(),
            grad_out.ptr<float>(),
            loss_scalar_.ptr<float>(),
            reduction_buffer_.ptr<float>(),
            H,
            W,
            nullptr);

        return loss_scalar_;
    }

    // ------------------------------
    // PUBLIC FORWARD METHODS
    // ------------------------------

    std::expected<std::pair<lfs::core::Tensor, DepthLoss::Context>, std::string>
    DepthLoss::forward(
        const lfs::core::Tensor& rendered_depth,
        const lfs::core::Tensor& gt_depth,
        const lfs::core::Tensor& valid_mask,
        const DepthLossConfig& config) {

        try {
            // Validate inputs
            if (rendered_depth.device() != lfs::core::Device::CUDA) {
                return std::unexpected("rendered_depth must be on CUDA device");
            }
            if (valid_mask.device() != lfs::core::Device::CUDA) {
                return std::unexpected("valid_mask must be on CUDA device");
            }
            if (rendered_depth.shape().rank() != 2) {
                return std::unexpected(std::format(
                    "rendered_depth must be 2D [H, W], got rank {}",
                    rendered_depth.shape().rank()));
            }
            if (rendered_depth.shape() != valid_mask.shape()) {
                return std::unexpected("rendered_depth and valid_mask must have same shape");
            }

            // Check if we have GT depth (supervised mode)
            bool has_gt_depth = gt_depth.is_valid() && gt_depth.numel() > 0;
            if (has_gt_depth) {
                if (gt_depth.device() != lfs::core::Device::CUDA) {
                    return std::unexpected("gt_depth must be on CUDA device");
                }
                if (rendered_depth.shape() != gt_depth.shape()) {
                    return std::unexpected("rendered_depth and gt_depth must have same shape");
                }
            }

            // Check configuration validity
            if (!config.use_l1 && !config.use_gradient_smoothness) {
                return std::unexpected("At least one of use_l1 or use_gradient_smoothness must be true");
            }
            if (config.use_l1 && !has_gt_depth) {
                return std::unexpected("L1 loss requires ground truth depth (gt_depth)");
            }

            size_t H = rendered_depth.size(0);
            size_t W = rendered_depth.size(1);
            size_t N = H * W;
            size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

            // Ensure buffers are allocated
            ensure_buffers(rendered_depth.shape().dims(), num_blocks);

            // Initialize gradient buffer
            auto grad_depth = lfs::core::Tensor::zeros({H, W}, lfs::core::Device::CUDA);

            lfs::core::Tensor total_loss = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            bool used_smoothness = false;

            // Compute L1 loss if enabled and GT is available
            if (config.use_l1 && has_gt_depth) {
                auto l1_result = compute_l1_loss(rendered_depth, gt_depth, valid_mask, grad_depth);
                if (!l1_result) {
                    return std::unexpected(std::format("L1 loss computation failed: {}", l1_result.error()));
                }

                // Scale by weight and add to total
                total_loss = total_loss + l1_result.value().mul(config.weight);
            }

            // Compute gradient smoothness if enabled
            if (config.use_gradient_smoothness) {
                auto smoothness_grad = lfs::core::Tensor::zeros({H, W}, lfs::core::Device::CUDA);
                auto smoothness_result = compute_smoothness_loss(rendered_depth, valid_mask, smoothness_grad);
                if (!smoothness_result) {
                    return std::unexpected(std::format("Smoothness loss computation failed: {}",
                                                       smoothness_result.error()));
                }

                // Scale by weight and add to total loss
                float combined_weight = config.weight * config.smoothness_weight;
                total_loss = total_loss + smoothness_result.value().mul(combined_weight);

                // Accumulate smoothness gradients
                grad_depth = grad_depth + smoothness_grad.mul(combined_weight);
                used_smoothness = true;
            }

            Context ctx{
                .loss_tensor = total_loss,
                .grad_depth = grad_depth,
                .has_gt_depth = has_gt_depth,
                .used_smoothness = used_smoothness};

            return std::make_pair(total_loss, ctx);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in DepthLoss::forward: {}", e.what()));
        }
    }

    std::expected<std::pair<lfs::core::Tensor, DepthLoss::Context>, std::string>
    DepthLoss::forward_edge_aware(
        const lfs::core::Tensor& rendered_depth,
        const lfs::core::Tensor& gt_depth,
        const lfs::core::Tensor& valid_mask,
        const lfs::core::Tensor& rendered_rgb,
        const DepthLossConfig& config) {

        try {
            // Validate inputs
            if (rendered_depth.device() != lfs::core::Device::CUDA) {
                return std::unexpected("rendered_depth must be on CUDA device");
            }
            if (valid_mask.device() != lfs::core::Device::CUDA) {
                return std::unexpected("valid_mask must be on CUDA device");
            }
            if (rendered_rgb.device() != lfs::core::Device::CUDA) {
                return std::unexpected("rendered_rgb must be on CUDA device");
            }
            if (rendered_depth.shape().rank() != 2) {
                return std::unexpected(std::format(
                    "rendered_depth must be 2D [H, W], got rank {}",
                    rendered_depth.shape().rank()));
            }
            if (rendered_rgb.shape().rank() != 3) {
                return std::unexpected(std::format(
                    "rendered_rgb must be 3D [H, W, C], got rank {}",
                    rendered_rgb.shape().rank()));
            }
            if (rendered_depth.size(0) != rendered_rgb.size(0) ||
                rendered_depth.size(1) != rendered_rgb.size(1)) {
                return std::unexpected("rendered_depth and rendered_rgb must have same H, W dimensions");
            }

            // Check if we have GT depth
            bool has_gt_depth = gt_depth.is_valid() && gt_depth.numel() > 0;
            if (has_gt_depth) {
                if (gt_depth.device() != lfs::core::Device::CUDA) {
                    return std::unexpected("gt_depth must be on CUDA device");
                }
                if (rendered_depth.shape() != gt_depth.shape()) {
                    return std::unexpected("rendered_depth and gt_depth must have same shape");
                }
            }

            size_t H = rendered_depth.size(0);
            size_t W = rendered_depth.size(1);
            size_t N = H * W;
            size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

            // Ensure buffers are allocated
            ensure_buffers(rendered_depth.shape().dims(), num_blocks);

            // Initialize gradient buffer
            auto grad_depth = lfs::core::Tensor::zeros({H, W}, lfs::core::Device::CUDA);

            lfs::core::Tensor total_loss = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            bool used_smoothness = false;

            // Compute L1 loss if enabled and GT is available
            if (config.use_l1 && has_gt_depth) {
                auto l1_result = compute_l1_loss(rendered_depth, gt_depth, valid_mask, grad_depth);
                if (!l1_result) {
                    return std::unexpected(std::format("L1 loss computation failed: {}", l1_result.error()));
                }
                total_loss = total_loss + l1_result.value().mul(config.weight);
            }

            // Always use edge-aware smoothness in this variant
            if (config.use_gradient_smoothness) {
                auto smoothness_grad = lfs::core::Tensor::zeros({H, W}, lfs::core::Device::CUDA);
                auto smoothness_result = compute_edge_aware_smoothness(
                    rendered_depth, valid_mask, rendered_rgb, smoothness_grad);
                if (!smoothness_result) {
                    return std::unexpected(std::format("Edge-aware smoothness computation failed: {}",
                                                       smoothness_result.error()));
                }

                float combined_weight = config.weight * config.smoothness_weight;
                total_loss = total_loss + smoothness_result.value().mul(combined_weight);
                grad_depth = grad_depth + smoothness_grad.mul(combined_weight);
                used_smoothness = true;
            }

            Context ctx{
                .loss_tensor = total_loss,
                .grad_depth = grad_depth,
                .has_gt_depth = has_gt_depth,
                .used_smoothness = used_smoothness};

            return std::make_pair(total_loss, ctx);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in DepthLoss::forward_edge_aware: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
