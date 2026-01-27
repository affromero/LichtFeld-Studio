/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused L1 depth loss computation with gradient and masking
     *
     * Computes in a single optimized pass:
     * - loss = sum(|rendered - gt| * mask) / sum(mask)
     * - grad = sign(rendered - gt) * mask / sum(mask)
     *
     * Invalid pixels (mask == 0) are excluded from the loss computation.
     *
     * @param rendered_depth Rendered depth map [H, W]
     * @param gt_depth Ground truth depth map [H, W]
     * @param valid_mask Binary validity mask [H, W] (1 = valid, 0 = invalid)
     * @param grad_out Output gradient w.r.t. rendered_depth [H, W]
     * @param loss_out Output scalar loss [1]
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (H*W+255)/256) elements)
     * @param H Image height
     * @param W Image width
     * @param stream CUDA stream
     */
    void launch_fused_depth_l1_loss(
        const float* rendered_depth,
        const float* gt_depth,
        const float* valid_mask,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t H,
        size_t W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Depth gradient smoothness loss with masking
     *
     * Computes total variation (TV) style smoothness loss:
     * - loss = sum((|dD/dx| + |dD/dy|) * mask) / sum(mask)
     *
     * Uses forward differences for gradient computation:
     * - dD/dx = D[i, j+1] - D[i, j]
     * - dD/dy = D[i+1, j] - D[i, j]
     *
     * Gradient computation:
     * - grad[i,j] = sign(D[i,j] - D[i,j-1]) - sign(D[i,j+1] - D[i,j])
     *             + sign(D[i,j] - D[i-1,j]) - sign(D[i+1,j] - D[i,j])
     *
     * @param depth Depth map [H, W]
     * @param valid_mask Binary validity mask [H, W]
     * @param grad_out Output gradient w.r.t. depth [H, W] (accumulated)
     * @param loss_out Output scalar loss [1]
     * @param temp_buffer Temporary buffer for partial sums
     * @param H Image height
     * @param W Image width
     * @param stream CUDA stream
     */
    void launch_depth_gradient_smoothness(
        const float* depth,
        const float* valid_mask,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t H,
        size_t W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Edge-aware depth smoothness loss using RGB guidance
     *
     * Uses image gradients to modulate depth smoothness:
     * - loss = sum(exp(-|dI/dx|) * |dD/dx| + exp(-|dI/dy|) * |dD/dy|) * mask / sum(mask)
     *
     * This allows sharp depth discontinuities at image edges while encouraging
     * smooth depth in homogeneous regions.
     *
     * @param depth Depth map [H, W]
     * @param valid_mask Binary validity mask [H, W]
     * @param rgb RGB image [H, W, 3] for edge guidance
     * @param grad_out Output gradient w.r.t. depth [H, W] (accumulated)
     * @param loss_out Output scalar loss [1]
     * @param temp_buffer Temporary buffer for partial sums
     * @param H Image height
     * @param W Image width
     * @param stream CUDA stream
     */
    void launch_depth_edge_aware_smoothness(
        const float* depth,
        const float* valid_mask,
        const float* rgb,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t H,
        size_t W,
        cudaStream_t stream = nullptr);

    /**
     * @brief Scale-invariant depth loss (for monocular depth estimation)
     *
     * Computes the scale-invariant loss from Eigen et al. 2014:
     * - d_i = log(rendered) - log(gt)
     * - loss = (1/n) * sum(d_i^2) - (lambda/n^2) * (sum(d_i))^2
     *
     * This loss is invariant to global scale, useful when GT depth
     * is from monocular estimation with unknown scale.
     *
     * @param rendered_depth Rendered depth map [H, W]
     * @param gt_depth Ground truth depth map [H, W]
     * @param valid_mask Binary validity mask [H, W]
     * @param grad_out Output gradient w.r.t. rendered_depth [H, W]
     * @param loss_out Output scalar loss [1]
     * @param temp_buffer Temporary buffer for reductions
     * @param H Image height
     * @param W Image width
     * @param lambda Scale factor (typically 0.5)
     * @param stream CUDA stream
     */
    void launch_scale_invariant_depth_loss(
        const float* rendered_depth,
        const float* gt_depth,
        const float* valid_mask,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t H,
        size_t W,
        float lambda = 0.5f,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
