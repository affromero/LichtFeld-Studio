/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace lfs::training::kernels {

    // =============================================================================
    // Camera intrinsics/extrinsics structure for GPU
    // =============================================================================

    struct CameraParams {
        float fx, fy;          // Focal lengths
        float cx, cy;          // Principal point
        float R[9];            // 3x3 rotation matrix (row-major)
        float t[3];            // Translation vector
        int width, height;     // Image dimensions
    };

    // =============================================================================
    // Multi-view warping kernels
    // =============================================================================

    /**
     * @brief Compute depth-induced homography warping from reference to neighbor
     *
     * For each pixel (u, v) in reference view:
     * 1. Unproject to 3D: X = depth * K_ref^-1 * [u, v, 1]^T
     * 2. Transform to neighbor: X' = R_rel * X + t_rel
     * 3. Project to neighbor: [u', v'] = K_neighbor * X' / X'_z
     *
     * @param ref_depth [H, W] depth map
     * @param ref_normal [H, W, 3] normal map (world space)
     * @param warped_u [H, W] output warped u coordinates in neighbor view
     * @param warped_v [H, W] output warped v coordinates in neighbor view
     * @param ref_cam Reference camera parameters
     * @param neighbor_cam Neighbor camera parameters
     * @param R_rel [9] relative rotation (row-major)
     * @param t_rel [3] relative translation
     * @param height Image height
     * @param width Image width
     * @param min_depth Minimum valid depth
     * @param stream CUDA stream
     */
    void launch_depth_warp_forward(
        const float* ref_depth,
        const float* ref_normal,
        float* warped_u,
        float* warped_v,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel,
        const float* t_rel,
        int height,
        int width,
        float min_depth,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute inverse warping from neighbor back to reference (for cycle consistency)
     *
     * Given warped coordinates (u', v') in neighbor view and neighbor depth,
     * warp back to reference view to compute cycle reprojection.
     *
     * @param warped_u [H, W] warped u coordinates in neighbor
     * @param warped_v [H, W] warped v coordinates in neighbor
     * @param neighbor_depth [H_n, W_n] neighbor depth map (for sampling)
     * @param cycle_u [H, W] output cycle-warped u coordinates
     * @param cycle_v [H, W] output cycle-warped v coordinates
     * @param ref_cam Reference camera parameters
     * @param neighbor_cam Neighbor camera parameters
     * @param R_rel_inv [9] inverse relative rotation
     * @param t_rel_inv [3] inverse relative translation
     * @param ref_height Reference image height
     * @param ref_width Reference image width
     * @param neighbor_height Neighbor image height
     * @param neighbor_width Neighbor image width
     * @param min_depth Minimum valid depth
     * @param stream CUDA stream
     */
    void launch_depth_warp_inverse(
        const float* warped_u,
        const float* warped_v,
        const float* neighbor_depth,
        float* cycle_u,
        float* cycle_v,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel_inv,
        const float* t_rel_inv,
        int ref_height,
        int ref_width,
        int neighbor_height,
        int neighbor_width,
        float min_depth,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute cycle reprojection error and confidence weights
     *
     * For each pixel:
     *   error = sqrt((u - cycle_u)^2 + (v - cycle_v)^2)
     *   weight = exp(-error) if error < threshold, else 0
     *
     * @param cycle_u [H, W] cycle-warped u coordinates
     * @param cycle_v [H, W] cycle-warped v coordinates
     * @param error_out [H, W] output reprojection error
     * @param weight_out [H, W] output confidence weights
     * @param height Image height
     * @param width Image width
     * @param error_threshold Max error for non-zero weight
     * @param stream CUDA stream
     */
    void launch_cycle_error_and_weights(
        const float* cycle_u,
        const float* cycle_v,
        float* error_out,
        float* weight_out,
        int height,
        int width,
        float error_threshold,
        cudaStream_t stream = nullptr);

    // =============================================================================
    // NCC (Normalized Cross-Correlation) kernel
    // =============================================================================

    /**
     * @brief Compute NCC patch matching score between reference and warped neighbor
     *
     * For each pixel, compute NCC in a (patch_size x patch_size) window:
     *   NCC = sum((I_ref - mean_ref) * (I_warped - mean_warped)) /
     *         sqrt(sum((I_ref - mean_ref)^2) * sum((I_warped - mean_warped)^2))
     *
     * @param ref_image [H, W, C] reference image
     * @param neighbor_image [H_n, W_n, C] neighbor image
     * @param warped_u [H, W] warped u coordinates
     * @param warped_v [H, W] warped v coordinates
     * @param ncc_out [H, W] output NCC scores (-1 to 1, higher is better)
     * @param ref_height Reference image height
     * @param ref_width Reference image width
     * @param neighbor_height Neighbor image height
     * @param neighbor_width Neighbor image width
     * @param channels Number of color channels
     * @param patch_size NCC window size (must be odd)
     * @param stream CUDA stream
     */
    void launch_ncc_matching(
        const float* ref_image,
        const float* neighbor_image,
        const float* warped_u,
        const float* warped_v,
        float* ncc_out,
        int ref_height,
        int ref_width,
        int neighbor_height,
        int neighbor_width,
        int channels,
        int patch_size,
        cudaStream_t stream = nullptr);

    // =============================================================================
    // Combined loss computation kernel
    // =============================================================================

    /**
     * @brief Fused multi-view loss computation with reduction
     *
     * Computes: L = photometric_weight * (1 - NCC) + geometric_weight * error
     * weighted by confidence: L_total = sum(weight * L) / sum(weight)
     *
     * @param reprojection_error [H, W] cycle reprojection error
     * @param ncc_scores [H, W] NCC scores
     * @param confidence_weights [H, W] confidence weights
     * @param loss_out [1] output scalar loss
     * @param partial_sums Temporary buffer for reduction (min(1024, (H*W+255)/256))
     * @param height Image height
     * @param width Image width
     * @param photometric_weight Weight for photometric term
     * @param geometric_weight Weight for geometric term
     * @param stream CUDA stream
     */
    void launch_multiview_loss_reduction(
        const float* reprojection_error,
        const float* ncc_scores,
        const float* confidence_weights,
        float* loss_out,
        float* partial_sums,
        int height,
        int width,
        float photometric_weight,
        float geometric_weight,
        cudaStream_t stream = nullptr);

    // =============================================================================
    // Gradient computation kernels
    // =============================================================================

    /**
     * @brief Compute gradients w.r.t. depth and normal
     *
     * Backpropagates the multi-view loss through the warping chain to compute
     * gradients for depth and normal maps.
     *
     * @param reprojection_error [H, W] cycle reprojection error
     * @param cycle_u [H, W] cycle-warped u coordinates
     * @param cycle_v [H, W] cycle-warped v coordinates
     * @param confidence_weights [H, W] confidence weights
     * @param ref_depth [H, W] reference depth
     * @param ref_normal [H, W, 3] reference normals
     * @param depth_grad [H, W] output gradient accumulator for depth
     * @param normal_grad [H, W, 3] output gradient accumulator for normals
     * @param ref_cam Reference camera parameters
     * @param neighbor_cam Neighbor camera parameters
     * @param R_rel [9] relative rotation
     * @param t_rel [3] relative translation
     * @param height Image height
     * @param width Image width
     * @param geometric_weight Loss weight for geometric term
     * @param total_weight Sum of confidence weights (for normalization)
     * @param stream CUDA stream
     */
    void launch_multiview_backward(
        const float* reprojection_error,
        const float* cycle_u,
        const float* cycle_v,
        const float* confidence_weights,
        const float* ref_depth,
        const float* ref_normal,
        float* depth_grad,
        float* normal_grad,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel,
        const float* t_rel,
        int height,
        int width,
        float geometric_weight,
        float total_weight,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
