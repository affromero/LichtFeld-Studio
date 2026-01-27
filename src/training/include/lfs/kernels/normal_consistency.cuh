/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Compute Gaussian normals from quaternions and scales
     *
     * The normal direction is the rotation matrix column corresponding to
     * the smallest scale (shortest axis of the ellipsoid).
     *
     * For each Gaussian i:
     *   1. Convert quaternion q_i to rotation matrix R_i
     *   2. Find j = argmin(scales_i[0], scales_i[1], scales_i[2])
     *   3. Normal n_i = R_i[:, j] (column j of rotation matrix)
     *
     * @param quaternions [N, 4] quaternions (w, x, y, z) - must be normalized
     * @param scales [N, 3] scale factors (linear space, NOT log)
     * @param normals_out [N, 3] output unit normals
     * @param N Number of Gaussians
     * @param stream CUDA stream
     */
    void launch_compute_gaussian_normals(
        const float* quaternions,
        const float* scales,
        float* normals_out,
        int N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for Gaussian normal computation
     *
     * Computes gradients w.r.t. quaternions and scales given gradients
     * w.r.t. the output normals.
     *
     * @param quaternions [N, 4] quaternions (w, x, y, z)
     * @param scales [N, 3] scale factors (linear space)
     * @param grad_normals [N, 3] gradient w.r.t. output normals
     * @param grad_quaternions [N, 4] output gradient w.r.t. quaternions (accumulated)
     * @param grad_scales [N, 3] output gradient w.r.t. scales (accumulated)
     * @param N Number of Gaussians
     * @param stream CUDA stream
     */
    void launch_compute_gaussian_normals_backward(
        const float* quaternions,
        const float* scales,
        const float* grad_normals,
        float* grad_quaternions,
        float* grad_scales,
        int N,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute surface normals from depth map using finite differences
     *
     * Uses Sobel-like finite differences to compute depth gradients, then
     * computes normals as n = normalize(dz/dx x dz/dy).
     *
     * The depth gradient is computed in camera space accounting for
     * perspective projection:
     *   dX/dx = z / fx  (horizontal world-space displacement per pixel)
     *   dY/dy = z / fy  (vertical world-space displacement per pixel)
     *
     * Invalid regions (depth <= 0 or at boundaries) get zero normals.
     *
     * @param depth_map [H, W] depth values
     * @param normals_out [H, W, 3] output unit normals (zero for invalid)
     * @param H Image height
     * @param W Image width
     * @param fx Camera focal length x
     * @param fy Camera focal length y
     * @param stream CUDA stream
     */
    void launch_compute_depth_normals(
        const float* depth_map,
        float* normals_out,
        int H,
        int W,
        float fx,
        float fy,
        cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for depth normal computation
     *
     * Computes gradient w.r.t. depth map given gradient w.r.t. normals.
     * Uses chain rule through the Sobel convolution and normalization.
     *
     * @param depth_map [H, W] depth values
     * @param normals [H, W, 3] computed normals (from forward pass)
     * @param grad_normals [H, W, 3] gradient w.r.t. output normals
     * @param grad_depth [H, W] output gradient w.r.t. depth (accumulated)
     * @param H Image height
     * @param W Image width
     * @param fx Camera focal length x
     * @param fy Camera focal length y
     * @param stream CUDA stream
     */
    void launch_compute_depth_normals_backward(
        const float* depth_map,
        const float* normals,
        const float* grad_normals,
        float* grad_depth,
        int H,
        int W,
        float fx,
        float fy,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute normal consistency loss per pixel
     *
     * For each pixel, computes:
     *   loss = Σ_k ω_k · (1 - n_k · ñ)
     *
     * where:
     *   n_k = Gaussian normal for Gaussian k at this pixel
     *   ñ   = depth-derived surface normal at this pixel
     *   ω_k = blend weight for Gaussian k
     *
     * @param gaussian_normals [N, 3] precomputed Gaussian normals
     * @param depth_normals [H, W, 3] precomputed depth normals
     * @param blend_weights [H, W, K] per-pixel blend weights
     * @param gaussian_indices [H, W, K] Gaussian indices per pixel (-1 for invalid)
     * @param loss_per_pixel [H, W] output loss per pixel
     * @param H Image height
     * @param W Image width
     * @param K Max Gaussians per pixel
     * @param stream CUDA stream
     */
    void launch_normal_consistency_forward(
        const float* gaussian_normals,
        const float* depth_normals,
        const float* blend_weights,
        const int* gaussian_indices,
        float* loss_per_pixel,
        int H,
        int W,
        int K,
        cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for normal consistency loss
     *
     * Computes gradients w.r.t. Gaussian normals given per-pixel loss gradients.
     *
     * For each Gaussian i, accumulates gradients from all pixels where it
     * contributes:
     *   grad_n_i += Σ_p ω_i(p) · (-ñ(p)) · grad_loss(p)
     *
     * Note: Gradients w.r.t. depth normals are typically not needed since
     * depth is usually detached from the computation graph.
     *
     * @param gaussian_normals [N, 3] precomputed Gaussian normals
     * @param depth_normals [H, W, 3] precomputed depth normals
     * @param blend_weights [H, W, K] per-pixel blend weights
     * @param gaussian_indices [H, W, K] Gaussian indices per pixel
     * @param grad_loss_per_pixel [H, W] gradient w.r.t. per-pixel loss
     * @param grad_gaussian_normals [N, 3] output gradient (accumulated)
     * @param N Number of Gaussians
     * @param H Image height
     * @param W Image width
     * @param K Max Gaussians per pixel
     * @param stream CUDA stream
     */
    void launch_normal_consistency_backward(
        const float* gaussian_normals,
        const float* depth_normals,
        const float* blend_weights,
        const int* gaussian_indices,
        const float* grad_loss_per_pixel,
        float* grad_gaussian_normals,
        int N,
        int H,
        int W,
        int K,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused forward + backward for normal consistency loss
     *
     * Single-pass kernel that computes loss and accumulates gradients
     * directly to quaternion and scale gradients, avoiding intermediate
     * storage for Gaussian normal gradients.
     *
     * This is more memory-efficient for training as it:
     * 1. Computes Gaussian normals on-the-fly
     * 2. Computes loss contribution
     * 3. Backpropagates through normal computation immediately
     *
     * @param quaternions [N, 4] raw quaternions
     * @param scales [N, 3] raw scales (log space)
     * @param depth_normals [H, W, 3] precomputed depth normals
     * @param blend_weights [H, W, K] per-pixel blend weights
     * @param gaussian_indices [H, W, K] Gaussian indices per pixel
     * @param loss_out [1] scalar loss output (accumulated via atomic add)
     * @param grad_quaternions [N, 4] gradient output (accumulated)
     * @param grad_scales [N, 3] gradient output (accumulated)
     * @param N Number of Gaussians
     * @param H Image height
     * @param W Image width
     * @param K Max Gaussians per pixel
     * @param weight Loss weight multiplier
     * @param stream CUDA stream
     */
    void launch_fused_normal_consistency(
        const float* quaternions,
        const float* scales,
        const float* depth_normals,
        const float* blend_weights,
        const int* gaussian_indices,
        float* loss_out,
        float* grad_quaternions,
        float* grad_scales,
        int N,
        int H,
        int W,
        int K,
        float weight,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
