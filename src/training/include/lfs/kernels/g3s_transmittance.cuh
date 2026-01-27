/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // =========================================================================
    // DEVICE FUNCTIONS (INLINED IN HEADER FOR CROSS-COMPILATION)
    // =========================================================================

    /**
     * @brief Compute per-Gaussian stochastic transmittance from G3S paper
     *
     * Implements Equation 12 from "Gaussian Geometry Guidance for Scene Reconstruction":
     *
     *   T_i(t) = {
     *       v_i(t),                      if t <= t*_i
     *       v_i(t*_i)^2 / v_i(t),        if t > t*_i
     *   }
     *
     * where:
     *   - t       = query depth along ray
     *   - t*_i    = peak depth (Gaussian center projected to ray)
     *   - v_i(t)  = vacancy at depth t
     *
     * This creates "stochastic solid" behavior:
     *   - Before peak: entering a probabilistic cloud
     *   - After peak: exiting through the cloud's center
     *
     * @param t Query depth along the ray
     * @param t_star Peak depth (depth of Gaussian center projected onto ray)
     * @param vacancy_at_t_star Vacancy computed at the peak depth v(t*)
     * @param vacancy_at_t Vacancy computed at the query depth v(t)
     * @return Stochastic transmittance in range [0, 1]
     */
    __device__ __forceinline__ float compute_stochastic_transmittance(
        float t,
        float t_star,
        float vacancy_at_t_star,
        float vacancy_at_t) {

        // Handle edge cases for numerical stability
        if (vacancy_at_t <= 0.0f) {
            return 0.0f;  // Fully occluded
        }
        if (vacancy_at_t_star <= 0.0f) {
            // If vacancy at peak is zero, the Gaussian is maximally opaque
            // Return 0 if we're past the peak, vacancy_at_t otherwise
            return (t > t_star) ? 0.0f : vacancy_at_t;
        }

        if (t <= t_star) {
            // Before or at peak: T(t) = v(t)
            return vacancy_at_t;
        } else {
            // After peak: T(t) = v(t*)^2 / v(t)
            // This ensures continuity at t = t*: v(t*)^2 / v(t*) = v(t*)
            float T = (vacancy_at_t_star * vacancy_at_t_star) / vacancy_at_t;
            // Clamp to valid transmittance range
            return fminf(T, 1.0f);
        }
    }

    /**
     * @brief Compute combined transmittance for multiple Gaussians at query depth
     *
     * For a ray passing through N Gaussians sorted by depth, the total
     * transmittance at depth t is the product of individual transmittances:
     *
     *   T(t) = prod_i T_i(t)
     *
     * @param t Query depth
     * @param t_stars Array of peak depths [N]
     * @param vacancies_at_t_star Array of vacancies at peak [N]
     * @param vacancies_at_t Array of vacancies at query depth [N]
     * @param N Number of Gaussians
     * @return Combined transmittance in range [0, 1]
     */
    __device__ __forceinline__ float compute_combined_transmittance(
        float t,
        const float* t_stars,
        const float* vacancies_at_t_star,
        const float* vacancies_at_t,
        int N) {

        float T = 1.0f;
        for (int i = 0; i < N; ++i) {
            float Ti = compute_stochastic_transmittance(
                t,
                t_stars[i],
                vacancies_at_t_star[i],
                vacancies_at_t[i]);
            T *= Ti;

            // Early exit if transmittance drops below threshold
            if (T < 1e-6f) {
                return 0.0f;
            }
        }
        return T;
    }

    // =========================================================================
    // LAUNCH FUNCTIONS
    // =========================================================================

    /**
     * @brief Compute per-Gaussian stochastic transmittance for all pixels
     *
     * For each pixel, iterates through Gaussians in depth order and computes
     * the accumulated stochastic transmittance from G3S.
     *
     * The kernel uses the tile-based indexing from gsplat rasterization:
     *   - tile_offsets[tile_id] gives the starting index in flatten_ids
     *   - flatten_ids contains Gaussian indices sorted by depth per tile
     *
     * Memory layout:
     *   - depths[C, M]: Per-gaussian depth from camera
     *   - t_stars[C, M]: Peak depth (Gaussian center projected to ray)
     *   - vacancies_at_peak[C, M]: Vacancy computed at t*
     *   - tile_offsets[C, tile_H, tile_W]: CSR-style offsets into flatten_ids
     *   - flatten_ids[n_isects]: Sorted Gaussian indices per tile
     *
     * @param depths Per-gaussian depths from camera [C, M]
     * @param t_stars Peak depths for each Gaussian-ray pair [C, M]
     * @param vacancies_at_peak Vacancy at peak depth [C, M]
     * @param tile_offsets Tile range offsets [C, tile_height, tile_width]
     * @param flatten_ids Sorted gaussian indices [n_isects]
     * @param transmittance_out Output per-pixel transmittance [C, H, W]
     * @param query_t Depth at which to evaluate transmittance
     * @param C Number of cameras
     * @param H Image height
     * @param W Image width
     * @param M Number of Gaussians (visible per camera)
     * @param tile_size Tile size (typically 16)
     * @param n_isects Total number of tile-gaussian intersections
     * @param stream CUDA stream
     */
    void launch_compute_ray_transmittance(
        const float* depths,
        const float* t_stars,
        const float* vacancies_at_peak,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        float* transmittance_out,
        float query_t,
        int C,
        int H,
        int W,
        int M,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute stochastic transmittance map with vacancy computation on-the-fly
     *
     * This is a more efficient variant that computes vacancy values on-the-fly
     * rather than requiring pre-computed vacancy arrays. Uses the Gaussian
     * parameters directly.
     *
     * For each pixel:
     *   1. Create ray from camera through pixel
     *   2. For each Gaussian intersecting the tile:
     *      a. Compute t* (peak depth) from Gaussian center
     *      b. Compute v(t*) vacancy at peak
     *      c. Compute v(query_t) vacancy at query depth
     *      d. Compute T_i using stochastic transmittance formula
     *   3. Accumulate T = prod_i T_i
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4] (w, x, y, z)
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param tile_offsets Tile range offsets [C, tile_height, tile_width]
     * @param flatten_ids Sorted gaussian indices [n_isects]
     * @param visible_indices Maps local to global Gaussian index [M], or nullptr
     * @param transmittance_out Output per-pixel transmittance [C, H, W]
     * @param query_t Depth at which to evaluate transmittance
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians per camera
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size (typically 16)
     * @param n_isects Total number of tile-gaussian intersections
     * @param stream CUDA stream
     */
    void launch_compute_ray_transmittance_fused(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* viewmats,
        const float* Ks,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        float* transmittance_out,
        float query_t,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Backward pass for stochastic transmittance computation
     *
     * Computes gradients w.r.t. Gaussian parameters given the gradient
     * of the loss w.r.t. transmittance output.
     *
     * Chain rule through:
     *   dL/d(means) = dL/dT * dT/dv * dv/d(means)
     *   dL/d(quats) = dL/dT * dT/dv * dv/d(quats)
     *   dL/d(scales) = dL/dT * dT/dv * dv/d(scales)
     *   dL/d(opacities) = dL/dT * dT/dv * dv/d(opacities)
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param tile_offsets Tile range offsets
     * @param flatten_ids Sorted gaussian indices
     * @param visible_indices Maps local to global index, or nullptr
     * @param grad_transmittance Gradient w.r.t. output [C, H, W]
     * @param grad_means Output gradient w.r.t. means [N, 3]
     * @param grad_quats Output gradient w.r.t. quaternions [N, 4]
     * @param grad_scales Output gradient w.r.t. scales [N, 3]
     * @param grad_opacities Output gradient w.r.t. opacities [N]
     * @param query_t Depth at which transmittance was evaluated
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size
     * @param n_isects Number of intersections
     * @param stream CUDA stream
     */
    void launch_compute_ray_transmittance_backward(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* viewmats,
        const float* Ks,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        const float* grad_transmittance,
        float* grad_means,
        float* grad_quats,
        float* grad_scales,
        float* grad_opacities,
        float query_t,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute per-pixel transmittance at rendered depth
     *
     * A convenience wrapper that evaluates transmittance at the rendered
     * depth for each pixel, useful for depth-based regularization.
     *
     * For each pixel p with rendered depth d(p):
     *   T(p) = stochastic_transmittance(d(p))
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param viewmats Camera view matrices [C, 4, 4]
     * @param Ks Camera intrinsics [C, 3, 3]
     * @param rendered_depths Per-pixel rendered depth [C, H, W]
     * @param tile_offsets Tile range offsets
     * @param flatten_ids Sorted gaussian indices
     * @param visible_indices Maps local to global index, or nullptr
     * @param transmittance_out Output per-pixel transmittance [C, H, W]
     * @param C Number of cameras
     * @param N Total number of Gaussians
     * @param M Number of visible Gaussians
     * @param H Image height
     * @param W Image width
     * @param tile_size Tile size
     * @param n_isects Number of intersections
     * @param stream CUDA stream
     */
    void launch_compute_transmittance_at_rendered_depth(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* viewmats,
        const float* Ks,
        const float* rendered_depths,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices,
        float* transmittance_out,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
