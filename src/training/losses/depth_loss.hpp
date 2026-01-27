/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief Configuration for depth supervision loss
     *
     * Supports both supervised (with GT depth) and unsupervised (smoothness only) modes.
     */
    struct DepthLossConfig {
        float weight = 1.0f;               ///< Overall loss weight
        bool use_l1 = true;                ///< Use L1 loss for depth supervision
        bool use_gradient_smoothness = false; ///< Penalize depth discontinuities
        float smoothness_weight = 0.1f;    ///< Weight for gradient smoothness term
        float min_depth = 0.01f;           ///< Minimum valid depth value
        float max_depth = 100.0f;          ///< Maximum valid depth value
    };

    /**
     * @brief G3S Depth Supervision Loss with optional gradient smoothness regularization
     *
     * Forward:
     *   - Supervised mode: loss = L1(rendered_depth, gt_depth) where valid_mask == 1
     *   - Unsupervised mode: loss = gradient_smoothness(rendered_depth)
     *   - Combined mode: loss = L1_loss + smoothness_weight * gradient_smoothness
     *
     * Gradient Smoothness:
     *   Penalizes depth gradients to encourage smooth depth maps while preserving edges.
     *   Uses image-guided edge-aware smoothness when RGB is available.
     *
     * This is a libtorch-free implementation that uses lfs::core::Tensor.
     * OPTIMIZED: Pre-allocates buffers to minimize allocation overhead per training step.
     */
    struct DepthLoss {
        struct Context {
            lfs::core::Tensor loss_tensor;      ///< [1] scalar loss on GPU (avoid sync!)
            lfs::core::Tensor grad_depth;       ///< [H, W] gradient w.r.t. rendered depth
            bool has_gt_depth = false;          ///< Whether GT depth was provided
            bool used_smoothness = false;       ///< Whether smoothness loss was computed
        };

        /**
         * @brief Compute depth supervision loss and gradient
         *
         * @param rendered_depth [H, W] rendered depth map from splatting
         * @param gt_depth [H, W] ground truth depth (optional, empty tensor if unsupervised)
         * @param valid_mask [H, W] binary mask where 1 = valid pixel (handles sky, far regions)
         * @param config Loss configuration parameters
         * @return (loss_tensor, context) or error - loss stays on GPU!
         *
         * @note For unsupervised mode, pass an empty gt_depth tensor.
         * @note Invalid pixels (mask == 0) are excluded from loss computation.
         */
        std::expected<std::pair<lfs::core::Tensor, Context>, std::string> forward(
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& gt_depth,
            const lfs::core::Tensor& valid_mask,
            const DepthLossConfig& config);

        /**
         * @brief Compute depth loss with image-guided edge-aware smoothness
         *
         * Uses the rendered RGB image to guide smoothness, allowing sharp depth
         * discontinuities at image edges while smoothing depth elsewhere.
         *
         * @param rendered_depth [H, W] rendered depth map
         * @param gt_depth [H, W] ground truth depth (optional)
         * @param valid_mask [H, W] validity mask
         * @param rendered_rgb [H, W, 3] rendered RGB image for edge guidance
         * @param config Loss configuration
         * @return (loss_tensor, context) or error
         */
        std::expected<std::pair<lfs::core::Tensor, Context>, std::string> forward_edge_aware(
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& gt_depth,
            const lfs::core::Tensor& valid_mask,
            const lfs::core::Tensor& rendered_rgb,
            const DepthLossConfig& config);

    private:
        // Pre-allocated buffers for loss computation (eliminates allocation churn)
        lfs::core::Tensor grad_buffer_;         ///< Reusable gradient buffer [H, W]
        lfs::core::Tensor loss_scalar_;         ///< Reusable scalar loss [1]
        lfs::core::Tensor reduction_buffer_;    ///< Reusable reduction buffer
        lfs::core::Tensor grad_x_buffer_;       ///< Gradient in x direction [H, W]
        lfs::core::Tensor grad_y_buffer_;       ///< Gradient in y direction [H, W]
        std::vector<size_t> allocated_shape_;   ///< Track allocated shape
        size_t allocated_num_blocks_ = 0;       ///< Track reduction buffer size

        /**
         * @brief Ensure buffers are sized correctly for the input shape
         * Only reallocates if shape changed.
         */
        void ensure_buffers(const std::vector<size_t>& shape, size_t num_blocks);

        /**
         * @brief Compute L1 depth loss between rendered and GT depth
         * @return L1 loss tensor on GPU
         */
        std::expected<lfs::core::Tensor, std::string> compute_l1_loss(
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& gt_depth,
            const lfs::core::Tensor& valid_mask,
            lfs::core::Tensor& grad_out);

        /**
         * @brief Compute gradient smoothness loss
         * @return Smoothness loss tensor on GPU
         */
        std::expected<lfs::core::Tensor, std::string> compute_smoothness_loss(
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& valid_mask,
            lfs::core::Tensor& grad_out);

        /**
         * @brief Compute edge-aware gradient smoothness using RGB guidance
         * @return Edge-aware smoothness loss tensor on GPU
         */
        std::expected<lfs::core::Tensor, std::string> compute_edge_aware_smoothness(
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& valid_mask,
            const lfs::core::Tensor& rendered_rgb,
            lfs::core::Tensor& grad_out);
    };

} // namespace lfs::training::losses
