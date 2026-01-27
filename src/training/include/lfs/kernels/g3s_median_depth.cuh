/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // =========================================================================
    // G3S MEDIAN DEPTH - BINARY SEARCH ALGORITHM
    // =========================================================================

    /**
     * @brief Configuration for G3S median depth binary search
     *
     * Default values from the G3S paper:
     *   - search_radius: 0.4 (half-width of initial interval)
     *   - num_iterations: 5 (binary search iterations)
     *   - segments_per_iteration: 8 (subdivisions per iteration)
     *
     * Resolution after N iterations: 2 * r / 8^N
     * With defaults: 2 * 0.4 / 8^5 = 0.8 / 32768 ~= 2.4e-5 units
     */
    struct G3SMedianDepthConfig {
        float search_radius = 0.4f;
        int num_iterations = 5;
        int segments_per_iteration = 8;

        // Transmittance threshold for "crossing"
        // Median is where T crosses 0.5 (50% transmittance)
        float target_transmittance = 0.5f;

        // Early termination threshold
        // If transmittance is always above this, pixel is likely sky/empty
        float min_valid_transmittance = 1e-6f;
    };

    // =========================================================================
    // LAUNCH FUNCTIONS
    // =========================================================================

    /**
     * @brief Compute G3S median depth using binary search algorithm
     *
     * Implements the binary search median depth from G3S paper:
     *
     * Algorithm:
     *   1. Initialize interval [t_init - r, t_init + r]
     *   2. For N iterations:
     *      a. Divide interval into K segments (K-1 sample points)
     *      b. Compute T(t) at each sample point using stochastic transmittance
     *      c. Find segment where T crosses target (0.5)
     *      d. Narrow interval to that segment
     *   3. Return midpoint as t_med
     *
     * The median depth is where stochastic transmittance = 0.5, representing
     * the depth where half the scene density has been traversed.
     *
     * Edge cases:
     *   - T always > 0.5: No solid surface hit (sky/empty) -> invalid
     *   - T always < 0.5: Very opaque scene, use interval start
     *   - T never exactly 0.5: Use interpolation within final segment
     *
     * @param means Gaussian centers [N, 3] in world coordinates
     * @param quats Quaternions [N, 4] in (w, x, y, z) format
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param init_depths Initial depth estimate [C, H, W] from standard 3DGS or RaDe-GS
     * @param tile_offsets Per-tile Gaussian ranges [C, tile_H, tile_W]
     * @param flatten_ids Sorted Gaussian indices per tile [n_isects]
     * @param visible_indices Maps local to global Gaussian index [M], or nullptr
     * @param viewmats Camera view matrices [C, 4, 4] (world-to-camera)
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param median_depths Output G3S median depth [C, H, W]
     * @param valid_mask Output validity mask [C, H, W] (true where T crosses 0.5)
     * @param config Binary search configuration
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size (typically 16)
     * @param n_isects Total number of tile-gaussian intersections
     * @param stream CUDA stream
     */
    void launch_g3s_median_depth_forward(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* init_depths,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        const float* viewmats,
        const float* Ks,
        float* median_depths,
        bool* valid_mask,
        const G3SMedianDepthConfig& config,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Simplified interface with default configuration
     *
     * Convenience wrapper using default G3S paper parameters:
     *   - search_radius: 0.4
     *   - num_iterations: 5
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param init_depths Initial depth estimate [C, H, W]
     * @param tile_offsets Per-tile Gaussian ranges
     * @param flatten_ids Sorted Gaussian indices
     * @param visible_indices Local to global index mapping, or nullptr
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param median_depths Output G3S median depth [C, H, W]
     * @param valid_mask Output validity mask [C, H, W]
     * @param search_radius Search radius around initial depth (default: 0.4)
     * @param num_iterations Binary search iterations (default: 5)
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size
     * @param n_isects Number of intersections
     * @param stream CUDA stream
     */
    void launch_g3s_median_depth_forward(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* init_depths,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        const float* viewmats,
        const float* Ks,
        float* median_depths,
        bool* valid_mask,
        float search_radius,
        int num_iterations,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for G3S median depth
     *
     * Computes gradients w.r.t. Gaussian parameters using the implicit
     * function theorem. The median depth t_med satisfies T(t_med) = 0.5,
     * so we can use:
     *
     *   d(t_med)/d(params) = -[dT/d(params)] / [dT/dt]
     *
     * evaluated at t = t_med.
     *
     * This backward pass also supports stop-gradient behavior (disabled by
     * default) if only the geometry loss gradient is needed.
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param median_depths Computed median depths [C, H, W]
     * @param valid_mask Validity mask [C, H, W]
     * @param tile_offsets Per-tile Gaussian ranges
     * @param flatten_ids Sorted Gaussian indices
     * @param visible_indices Local to global index mapping, or nullptr
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param grad_median_depths Gradient w.r.t. median depths [C, H, W]
     * @param grad_means Output gradient w.r.t. means [N, 3]
     * @param grad_quats Output gradient w.r.t. quaternions [N, 4]
     * @param grad_scales Output gradient w.r.t. scales [N, 3]
     * @param grad_opacities Output gradient w.r.t. opacities [N]
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size
     * @param n_isects Number of intersections
     * @param stream CUDA stream
     */
    void launch_g3s_median_depth_backward(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* median_depths,
        const bool* valid_mask,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        const float* viewmats,
        const float* Ks,
        const float* grad_median_depths,
        float* grad_means,
        float* grad_quats,
        float* grad_scales,
        float* grad_opacities,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Complete closed-form backward pass for G3S median depth
     *
     * This is the full implementation of Equation 13 from the G3S paper:
     *
     *   d(t_med)/d(theta) = -[dT(t_med; theta)/d(theta)] / [dT(t; theta)/dt]|_{t=t_med}
     *
     * Key insight from the paper: The gradient distributes to ALL Gaussians
     * along the ray, not just the one that "won" the median depth. This is
     * why G3S produces better geometry - every Gaussian gets supervision signal.
     *
     * This version computes gradients for ALL parameters:
     *   - means: through vacancy -> Mahalanobis distance -> delta
     *   - quats: through vacancy -> Mahalanobis distance -> precision -> rotation
     *   - scales: through vacancy -> Mahalanobis distance -> precision -> scale
     *   - opacities: through vacancy -> occupancy -> opacity
     *
     * The implementation uses:
     *   1. Finite differences for dT/dt (more stable than analytical)
     *   2. Analytical gradients for dT/d(params) through the chain rule
     *   3. Atomic operations for gradient accumulation (many pixels per Gaussian)
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param median_depths Computed median depths [C, H, W]
     * @param valid_mask Validity mask [C, H, W]
     * @param tile_offsets Per-tile Gaussian ranges
     * @param flatten_ids Sorted Gaussian indices
     * @param visible_indices Local to global index mapping, or nullptr
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param grad_median_depths Gradient w.r.t. median depths [C, H, W]
     * @param grad_means Output gradient w.r.t. means [N, 3]
     * @param grad_quats Output gradient w.r.t. quaternions [N, 4]
     * @param grad_scales Output gradient w.r.t. scales [N, 3]
     * @param grad_opacities Output gradient w.r.t. opacities [N]
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size
     * @param n_isects Number of intersections
     * @param stream CUDA stream
     */
    void launch_g3s_median_depth_backward_complete(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* median_depths,
        const bool* valid_mask,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        const float* viewmats,
        const float* Ks,
        const float* grad_median_depths,
        float* grad_means,
        float* grad_quats,
        float* grad_scales,
        float* grad_opacities,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
