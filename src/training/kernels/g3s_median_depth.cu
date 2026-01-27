/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/g3s_median_depth.cuh"
#include "lfs/kernels/g3s_transmittance.cuh"
#include <cooperative_groups.h>

namespace lfs::training::kernels {

    namespace cg = cooperative_groups;

    // =========================================================================
    // DEVICE HELPER FUNCTIONS
    // =========================================================================

    /**
     * @brief Convert quaternion to 3x3 rotation matrix
     *
     * Quaternion format: (w, x, y, z)
     * Returns column-major rotation matrix stored in 9 floats.
     */
    __device__ __forceinline__ void quat_to_rotmat_median(
        float w, float x, float y, float z,
        float* R) {

        // Normalize quaternion
        float inv_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;

        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        // Column-major: R[col * 3 + row]
        // Column 0
        R[0] = 1.0f - 2.0f * (y2 + z2);
        R[1] = 2.0f * (xy + wz);
        R[2] = 2.0f * (xz - wy);
        // Column 1
        R[3] = 2.0f * (xy - wz);
        R[4] = 1.0f - 2.0f * (x2 + z2);
        R[5] = 2.0f * (yz + wx);
        // Column 2
        R[6] = 2.0f * (xz + wy);
        R[7] = 2.0f * (yz - wx);
        R[8] = 1.0f - 2.0f * (x2 + y2);
    }

    /**
     * @brief Compute inverse covariance (precision) matrix from quaternion and log-scales
     *
     * Returns the upper triangular part of the symmetric 3x3 precision matrix:
     *   [prec[0], prec[1], prec[2]]
     *   [prec[1], prec[3], prec[4]]
     *   [prec[2], prec[4], prec[5]]
     */
    __device__ __forceinline__ void compute_precision_matrix_median(
        float qw, float qx, float qy, float qz,
        float log_sx, float log_sy, float log_sz,
        float* prec) {

        float R[9];
        quat_to_rotmat_median(qw, qx, qy, qz, R);

        // Compute inverse scales squared: 1/s^2 = exp(-2 * log_s)
        float inv_s0_sq = expf(-2.0f * log_sx);
        float inv_s1_sq = expf(-2.0f * log_sy);
        float inv_s2_sq = expf(-2.0f * log_sz);

        // Precision = R * S^-2 * R^T
        prec[0] = R[0] * R[0] * inv_s0_sq + R[3] * R[3] * inv_s1_sq + R[6] * R[6] * inv_s2_sq;
        prec[1] = R[0] * R[1] * inv_s0_sq + R[3] * R[4] * inv_s1_sq + R[6] * R[7] * inv_s2_sq;
        prec[2] = R[0] * R[2] * inv_s0_sq + R[3] * R[5] * inv_s1_sq + R[6] * R[8] * inv_s2_sq;
        prec[3] = R[1] * R[1] * inv_s0_sq + R[4] * R[4] * inv_s1_sq + R[7] * R[7] * inv_s2_sq;
        prec[4] = R[1] * R[2] * inv_s0_sq + R[4] * R[5] * inv_s1_sq + R[7] * R[8] * inv_s2_sq;
        prec[5] = R[2] * R[2] * inv_s0_sq + R[5] * R[5] * inv_s1_sq + R[8] * R[8] * inv_s2_sq;
    }

    /**
     * @brief Compute Mahalanobis distance squared
     */
    __device__ __forceinline__ float mahalanobis_sq_median(
        float dx, float dy, float dz,
        const float* prec) {

        return prec[0] * dx * dx +
               2.0f * prec[1] * dx * dy +
               2.0f * prec[2] * dx * dz +
               prec[3] * dy * dy +
               2.0f * prec[4] * dy * dz +
               prec[5] * dz * dz;
    }

    /**
     * @brief Sigmoid activation function
     */
    __device__ __forceinline__ float sigmoid_median(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    /**
     * @brief Compute vacancy at a 3D point for a single Gaussian
     */
    __device__ __forceinline__ float compute_vacancy_at_point_median(
        float px, float py, float pz,
        float mx, float my, float mz,
        const float* prec,
        float opacity) {

        float dx = px - mx;
        float dy = py - my;
        float dz = pz - mz;

        float maha_sq = mahalanobis_sq_median(dx, dy, dz, prec);
        float exponent = fminf(maha_sq * 0.5f, 88.0f);
        float occupancy = opacity * expf(-exponent);

        return sqrtf(fmaxf(0.0f, 1.0f - occupancy));
    }

    /**
     * @brief Compute peak depth t* for a Gaussian along a ray
     *
     * The peak depth is where the Mahalanobis distance is minimized along the ray.
     * For ray r(t) = o + t*d:
     *   t* = (d^T * P * (mu - o)) / (d^T * P * d)
     */
    __device__ __forceinline__ float compute_peak_depth_median(
        float ox, float oy, float oz,
        float dx, float dy, float dz,
        float mx, float my, float mz,
        const float* prec) {

        float diff_x = mx - ox;
        float diff_y = my - oy;
        float diff_z = mz - oz;

        // d^T * P * d
        float dPd = prec[0] * dx * dx +
                    2.0f * prec[1] * dx * dy +
                    2.0f * prec[2] * dx * dz +
                    prec[3] * dy * dy +
                    2.0f * prec[4] * dy * dz +
                    prec[5] * dz * dz;

        // P * (mu - o)
        float Pd_x = prec[0] * diff_x + prec[1] * diff_y + prec[2] * diff_z;
        float Pd_y = prec[1] * diff_x + prec[3] * diff_y + prec[4] * diff_z;
        float Pd_z = prec[2] * diff_x + prec[4] * diff_y + prec[5] * diff_z;

        // d^T * (P * (mu - o))
        float dPdiff = dx * Pd_x + dy * Pd_y + dz * Pd_z;

        return dPdiff / fmaxf(dPd, 1e-10f);
    }

    /**
     * @brief Create ray from pixel coordinates
     */
    __device__ __forceinline__ void pixel_to_ray_median(
        float px, float py,
        const float* viewmat,
        const float* K,
        float& ox, float& oy, float& oz,
        float& dx, float& dy, float& dz) {

        // Camera intrinsics (row-major K matrix)
        float fx = K[0];
        float fy = K[4];
        float cx_K = K[2];
        float cy_K = K[5];

        // Ray direction in camera space (normalized)
        float ray_cam_x = (px - cx_K) / fx;
        float ray_cam_y = (py - cy_K) / fy;
        float ray_cam_z = 1.0f;

        float norm = rsqrtf(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z);
        ray_cam_x *= norm;
        ray_cam_y *= norm;
        ray_cam_z *= norm;

        // Camera-to-world rotation (transpose of world-to-camera)
        dx = viewmat[0] * ray_cam_x + viewmat[1] * ray_cam_y + viewmat[2] * ray_cam_z;
        dy = viewmat[4] * ray_cam_x + viewmat[5] * ray_cam_y + viewmat[6] * ray_cam_z;
        dz = viewmat[8] * ray_cam_x + viewmat[9] * ray_cam_y + viewmat[10] * ray_cam_z;

        // Camera position in world space: -R^T * t
        float tx = viewmat[12];
        float ty = viewmat[13];
        float tz = viewmat[14];
        ox = -(viewmat[0] * tx + viewmat[1] * ty + viewmat[2] * tz);
        oy = -(viewmat[4] * tx + viewmat[5] * ty + viewmat[6] * tz);
        oz = -(viewmat[8] * tx + viewmat[9] * ty + viewmat[10] * tz);
    }

    // =========================================================================
    // TRANSMITTANCE COMPUTATION FOR BINARY SEARCH
    // =========================================================================

    /**
     * @brief Compute stochastic transmittance at a query depth for all tile Gaussians
     *
     * This is the core function used during binary search. It computes the
     * accumulated transmittance T(t) at depth t by iterating through all
     * Gaussians in the tile.
     *
     * @param query_t Query depth along the ray
     * @param ray_ox, ray_oy, ray_oz Ray origin (camera position)
     * @param ray_dx, ray_dy, ray_dz Ray direction (normalized)
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param flatten_ids Sorted Gaussian indices for this tile
     * @param visible_indices Local to global mapping (or nullptr)
     * @param range_start Start index in flatten_ids for this tile
     * @param range_end End index in flatten_ids for this tile
     * @return Accumulated stochastic transmittance at query_t
     */
    __device__ float compute_transmittance_at_depth(
        float query_t,
        float ray_ox, float ray_oy, float ray_oz,
        float ray_dx, float ray_dy, float ray_dz,
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        int32_t range_start,
        int32_t range_end) {

        float T = 1.0f;

        for (int32_t idx = range_start; idx < range_end; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

            // Load Gaussian parameters
            float mx = means[global_g * 3 + 0];
            float my = means[global_g * 3 + 1];
            float mz = means[global_g * 3 + 2];

            float qw = quats[global_g * 4 + 0];
            float qx = quats[global_g * 4 + 1];
            float qy = quats[global_g * 4 + 2];
            float qz = quats[global_g * 4 + 3];

            float log_sx = scales[global_g * 3 + 0];
            float log_sy = scales[global_g * 3 + 1];
            float log_sz = scales[global_g * 3 + 2];

            float opacity = sigmoid_median(opacities[global_g]);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix_median(qw, qx, qy, qz, log_sx, log_sy, log_sz, prec);

            // Compute peak depth t*
            float t_star = compute_peak_depth_median(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec);

            // Skip if t* is behind camera
            if (t_star < 0.0f) continue;

            // Compute vacancy at t*
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;

            float vacancy_at_t_star = compute_vacancy_at_point_median(
                pt_star_x, pt_star_y, pt_star_z,
                mx, my, mz,
                prec, opacity);

            // Compute vacancy at query_t
            float pt_t_x = ray_ox + query_t * ray_dx;
            float pt_t_y = ray_oy + query_t * ray_dy;
            float pt_t_z = ray_oz + query_t * ray_dz;

            float vacancy_at_t = compute_vacancy_at_point_median(
                pt_t_x, pt_t_y, pt_t_z,
                mx, my, mz,
                prec, opacity);

            // Compute stochastic transmittance for this Gaussian
            float Ti = compute_stochastic_transmittance(
                query_t, t_star, vacancy_at_t_star, vacancy_at_t);

            T *= Ti;

            // Early exit if transmittance is effectively zero
            if (T < 1e-8f) {
                return 0.0f;
            }
        }

        return T;
    }

    // =========================================================================
    // FORWARD KERNEL: BINARY SEARCH MEDIAN DEPTH
    // =========================================================================

    /**
     * @brief Kernel for computing G3S median depth using binary search
     *
     * Each thread handles one pixel. Uses the following algorithm:
     *
     * 1. Initialize search interval [t_init - r, t_init + r]
     * 2. For each iteration:
     *    a. Divide interval into K segments (sample at K-1 interior points)
     *    b. Compute transmittance T(t) at each sample point
     *    c. Find the segment where T crosses the target (0.5)
     *    d. Narrow the interval to that segment
     * 3. Use linear interpolation within final segment for sub-segment precision
     * 4. Mark pixel as invalid if T never crosses 0.5
     */
    __global__ void g3s_median_depth_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ init_depths,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        const float* __restrict__ viewmats,
        const float* __restrict__ Ks,
        float* __restrict__ median_depths,
        bool* __restrict__ valid_mask,
        float search_radius,
        int num_iterations,
        int segments_per_iteration,
        float target_transmittance,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects) {

        auto block = cg::this_thread_block();
        int32_t cid = block.group_index().x;

        // Compute tile dimensions
        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;
        int tile_id = block.group_index().y * tile_width + block.group_index().z;

        // Pixel coordinates
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
        float px = (float)j + 0.5f;
        float py = (float)i + 0.5f;

        // Return if out of bounds
        if (i >= H || j >= W) return;

        int pix_id = i * W + j;
        int global_pix_id = cid * H * W + pix_id;

        // Get initial depth estimate
        float t_init = init_depths[global_pix_id];

        // Handle invalid initial depth
        if (t_init <= 0.0f || !isfinite(t_init)) {
            median_depths[global_pix_id] = 0.0f;
            valid_mask[global_pix_id] = false;
            return;
        }

        // Shift pointers for current camera
        const int32_t* tile_offsets_cam = tile_offsets + cid * tile_height * tile_width;
        const float* viewmat = viewmats + cid * 16;
        const float* K = Ks + cid * 9;

        // Get range of Gaussians for this tile
        int32_t range_start = tile_offsets_cam[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets_cam[tile_id + 1];

        // Handle empty tiles
        if (range_start >= range_end) {
            median_depths[global_pix_id] = t_init;
            valid_mask[global_pix_id] = false;
            return;
        }

        // Create ray for this pixel
        float ray_ox, ray_oy, ray_oz;
        float ray_dx, ray_dy, ray_dz;
        pixel_to_ray_median(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Initialize search interval
        float t_min = fmaxf(t_init - search_radius, 1e-4f);  // Don't go behind camera
        float t_max = t_init + search_radius;

        // Check transmittance at interval endpoints to verify crossing exists
        float T_min = compute_transmittance_at_depth(
            t_min, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            means, quats, scales, opacities,
            flatten_ids, visible_indices, range_start, range_end);

        float T_max = compute_transmittance_at_depth(
            t_max, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            means, quats, scales, opacities,
            flatten_ids, visible_indices, range_start, range_end);

        // Check if target crossing exists
        // T should decrease along ray (from 1.0 towards 0.0)
        bool crossing_exists = (T_min >= target_transmittance && T_max <= target_transmittance) ||
                               (T_min <= target_transmittance && T_max >= target_transmittance);

        // Handle edge cases
        if (!crossing_exists) {
            if (T_min < target_transmittance && T_max < target_transmittance) {
                // Very opaque - crossing happened before interval
                // Use interval start as best estimate
                median_depths[global_pix_id] = t_min;
                valid_mask[global_pix_id] = false;
            } else if (T_min > target_transmittance && T_max > target_transmittance) {
                // Transparent - no solid surface
                // This is the "sky pixel" case
                median_depths[global_pix_id] = t_init;
                valid_mask[global_pix_id] = false;
            } else {
                // Edge case - use initial estimate
                median_depths[global_pix_id] = t_init;
                valid_mask[global_pix_id] = false;
            }
            return;
        }

        // Binary search iterations
        // We maintain [t_min, t_max] as the current search interval
        // and [T_min, T_max] as transmittance at the endpoints
        for (int iter = 0; iter < num_iterations; ++iter) {
            float segment_width = (t_max - t_min) / (float)segments_per_iteration;

            // Sample transmittance at segment boundaries
            // We need segments_per_iteration + 1 sample points for segments_per_iteration segments
            // But we already have T_min and T_max, so we compute interior points

            // Storage for sample transmittances (on stack, small fixed size)
            // segments_per_iteration is typically 8, so max 9 samples
            float T_samples[9];
            float t_samples[9];

            T_samples[0] = T_min;
            t_samples[0] = t_min;

            // Compute interior samples
            for (int s = 1; s < segments_per_iteration; ++s) {
                float t_sample = t_min + s * segment_width;
                t_samples[s] = t_sample;
                T_samples[s] = compute_transmittance_at_depth(
                    t_sample, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
                    means, quats, scales, opacities,
                    flatten_ids, visible_indices, range_start, range_end);
            }

            T_samples[segments_per_iteration] = T_max;
            t_samples[segments_per_iteration] = t_max;

            // Find segment where crossing occurs
            // Looking for segment where T goes from >= target to < target
            // (or vice versa depending on monotonicity direction)
            int crossing_segment = -1;

            for (int s = 0; s < segments_per_iteration; ++s) {
                float T_left = T_samples[s];
                float T_right = T_samples[s + 1];

                // Check if target is within this segment
                bool contains_target = (T_left >= target_transmittance && T_right <= target_transmittance) ||
                                       (T_left <= target_transmittance && T_right >= target_transmittance);

                if (contains_target) {
                    crossing_segment = s;
                    break;
                }
            }

            // Update interval to crossing segment
            if (crossing_segment >= 0) {
                t_min = t_samples[crossing_segment];
                t_max = t_samples[crossing_segment + 1];
                T_min = T_samples[crossing_segment];
                T_max = T_samples[crossing_segment + 1];
            } else {
                // No crossing found - shouldn't happen if crossing_exists was true
                // Use midpoint as fallback
                break;
            }
        }

        // Linear interpolation within final segment for sub-segment precision
        // Find t where T(t) = target_transmittance using linear interpolation
        float t_med;
        if (fabsf(T_max - T_min) > 1e-8f) {
            // Interpolate: t = t_min + (target - T_min) / (T_max - T_min) * (t_max - t_min)
            float alpha = (target_transmittance - T_min) / (T_max - T_min);
            alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);  // Clamp to [0, 1]
            t_med = t_min + alpha * (t_max - t_min);
        } else {
            // Flat transmittance - use midpoint
            t_med = 0.5f * (t_min + t_max);
        }

        // Output results
        median_depths[global_pix_id] = t_med;
        valid_mask[global_pix_id] = true;
    }

    // =========================================================================
    // BACKWARD KERNEL
    // =========================================================================

    /**
     * @brief Backward pass for G3S median depth
     *
     * Uses implicit function theorem to compute gradients.
     * If T(t_med, params) = target, then:
     *   d(t_med)/d(params) = -[dT/d(params)] / [dT/dt]
     *
     * This requires computing:
     *   1. dT/dt at t_med (numerical derivative)
     *   2. dT/d(params) at t_med (analytical gradient through transmittance)
     */
    __global__ void g3s_median_depth_backward_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ median_depths,
        const bool* __restrict__ valid_mask,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        const float* __restrict__ viewmats,
        const float* __restrict__ Ks,
        const float* __restrict__ grad_median_depths,
        float* __restrict__ grad_means,
        float* __restrict__ grad_quats,
        float* __restrict__ grad_scales,
        float* __restrict__ grad_opacities,
        int C,
        int N,
        int M,
        int H,
        int W,
        int tile_size,
        int n_isects) {

        auto block = cg::this_thread_block();
        int32_t cid = block.group_index().x;

        // Compute tile dimensions
        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;
        int tile_id = block.group_index().y * tile_width + block.group_index().z;

        // Pixel coordinates
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
        float px = (float)j + 0.5f;
        float py = (float)i + 0.5f;

        if (i >= H || j >= W) return;

        int pix_id = i * W + j;
        int global_pix_id = cid * H * W + pix_id;

        // Skip invalid pixels
        if (!valid_mask[global_pix_id]) return;

        float grad_t_med = grad_median_depths[global_pix_id];
        if (fabsf(grad_t_med) < 1e-10f) return;

        float t_med = median_depths[global_pix_id];

        // Get camera data
        const int32_t* tile_offsets_cam = tile_offsets + cid * tile_height * tile_width;
        const float* viewmat = viewmats + cid * 16;
        const float* K = Ks + cid * 9;

        int32_t range_start = tile_offsets_cam[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets_cam[tile_id + 1];

        if (range_start >= range_end) return;

        // Create ray
        float ray_ox, ray_oy, ray_oz;
        float ray_dx, ray_dy, ray_dz;
        pixel_to_ray_median(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Compute dT/dt at t_med using finite differences
        float epsilon = 1e-4f;
        float T_plus = compute_transmittance_at_depth(
            t_med + epsilon, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            means, quats, scales, opacities,
            flatten_ids, visible_indices, range_start, range_end);

        float T_minus = compute_transmittance_at_depth(
            t_med - epsilon, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            means, quats, scales, opacities,
            flatten_ids, visible_indices, range_start, range_end);

        float dT_dt = (T_plus - T_minus) / (2.0f * epsilon);

        // Avoid division by zero
        if (fabsf(dT_dt) < 1e-10f) return;

        // Compute gradients for each Gaussian using chain rule
        // dL/d(params) = dL/d(t_med) * d(t_med)/d(params)
        //              = dL/d(t_med) * (-1/dT_dt) * dT/d(params)
        float grad_scale = -grad_t_med / dT_dt;

        // Compute point on ray at t_med
        float pt_x = ray_ox + t_med * ray_dx;
        float pt_y = ray_oy + t_med * ray_dy;
        float pt_z = ray_oz + t_med * ray_dz;

        // Iterate through Gaussians and compute gradients
        for (int32_t idx = range_start; idx < range_end; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

            // Load Gaussian parameters
            float mx = means[global_g * 3 + 0];
            float my = means[global_g * 3 + 1];
            float mz = means[global_g * 3 + 2];

            float qw = quats[global_g * 4 + 0];
            float qx_q = quats[global_g * 4 + 1];
            float qy_q = quats[global_g * 4 + 2];
            float qz_q = quats[global_g * 4 + 3];

            float log_sx = scales[global_g * 3 + 0];
            float log_sy = scales[global_g * 3 + 1];
            float log_sz = scales[global_g * 3 + 2];

            float raw_opacity = opacities[global_g];
            float opacity = sigmoid_median(raw_opacity);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix_median(qw, qx_q, qy_q, qz_q, log_sx, log_sy, log_sz, prec);

            // Compute peak depth t*
            float t_star = compute_peak_depth_median(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec);

            if (t_star < 0.0f) continue;

            // Compute vacancy values
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;

            float vacancy_at_t_star = compute_vacancy_at_point_median(
                pt_star_x, pt_star_y, pt_star_z,
                mx, my, mz,
                prec, opacity);

            float vacancy_at_t = compute_vacancy_at_point_median(
                pt_x, pt_y, pt_z,
                mx, my, mz,
                prec, opacity);

            // Skip if vacancies are too small (numerical stability)
            if (vacancy_at_t < 1e-6f || vacancy_at_t_star < 1e-6f) continue;

            // Compute gradient contribution through vacancy -> opacity
            // vacancy = sqrt(1 - o * G) where G = exp(-0.5 * maha)
            // d(vacancy)/d(opacity) = -G / (2 * vacancy)

            float dx = pt_x - mx;
            float dy = pt_y - my;
            float dz = pt_z - mz;
            float maha_sq = mahalanobis_sq_median(dx, dy, dz, prec);
            float G = expf(-fminf(maha_sq * 0.5f, 88.0f));

            float dv_do = -G / (2.0f * vacancy_at_t);

            // Transmittance contribution depends on whether t_med < t_star or >= t_star
            // For t <= t_star: T_i = vacancy_at_t, dT_i/d(vacancy) = 1
            // For t > t_star: T_i = v*^2/v_t, dT_i/d(v_t) = -v*^2/v_t^2, dT_i/d(v*) = 2v*/v_t
            float dT_dv_t = 0.0f;
            if (t_med <= t_star) {
                dT_dv_t = 1.0f;
            } else {
                dT_dv_t = -(vacancy_at_t_star * vacancy_at_t_star) / (vacancy_at_t * vacancy_at_t);
            }

            // Chain rule: dT/d(opacity) = dT/d(vacancy) * d(vacancy)/d(opacity)
            float dT_do = dT_dv_t * dv_do;

            // Final gradient contribution
            float grad_opacity_contrib = grad_scale * dT_do;

            // Sigmoid backward: d(sigmoid)/d(raw) = sigmoid * (1 - sigmoid)
            float grad_raw_opacity = grad_opacity_contrib * opacity * (1.0f - opacity);

            // Atomic add to gradient
            atomicAdd(&grad_opacities[global_g], grad_raw_opacity);

            // Note: Full implementation would also compute gradients for means, quats, scales
            // This requires backprop through the precision matrix and Mahalanobis distance
            // For brevity, we focus on opacity gradient which is the primary driver
        }
    }

    // =========================================================================
    // LAUNCH FUNCTIONS
    // =========================================================================

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0 || N == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        g3s_median_depth_kernel<<<grid, threads, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            init_depths,
            tile_offsets,
            flatten_ids,
            visible_indices,
            viewmats,
            Ks,
            median_depths,
            valid_mask,
            config.search_radius,
            config.num_iterations,
            config.segments_per_iteration,
            config.target_transmittance,
            C, N, M, H, W,
            tile_size,
            n_isects);
    }

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
        cudaStream_t stream) {

        G3SMedianDepthConfig config;
        config.search_radius = search_radius;
        config.num_iterations = num_iterations;

        launch_g3s_median_depth_forward(
            means, quats, scales, opacities,
            init_depths, tile_offsets, flatten_ids, visible_indices,
            viewmats, Ks, median_depths, valid_mask,
            config, C, N, M, H, W, tile_size, n_isects, stream);
    }

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
        cudaStream_t stream) {

        // Delegate to the complete backward implementation
        // which computes gradients for ALL Gaussian parameters
        launch_g3s_median_depth_backward_complete(
            means, quats, scales, opacities,
            median_depths, valid_mask,
            tile_offsets, flatten_ids, visible_indices,
            viewmats, Ks,
            grad_median_depths,
            grad_means, grad_quats, grad_scales, grad_opacities,
            C, N, M, H, W, tile_size, n_isects, stream);
    }

} // namespace lfs::training::kernels
