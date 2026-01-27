/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

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
    __device__ __forceinline__ void quat_to_rotmat(
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
     * @brief Compute inverse covariance matrix from quaternion and log-scales
     *
     * Covariance: Sigma = R * S^2 * R^T
     * Inverse:    Sigma^-1 = R * S^-2 * R^T
     *
     * Returns the upper triangular part of the symmetric 3x3 precision matrix:
     *   [prec[0], prec[1], prec[2]]
     *   [prec[1], prec[3], prec[4]]
     *   [prec[2], prec[4], prec[5]]
     */
    __device__ __forceinline__ void compute_precision_matrix(
        float qw, float qx, float qy, float qz,
        float log_sx, float log_sy, float log_sz,
        float* prec) {

        // Get rotation matrix
        float R[9];
        quat_to_rotmat(qw, qx, qy, qz, R);

        // Compute inverse scales squared: 1/s^2 = exp(-2 * log_s)
        float inv_s0_sq = expf(-2.0f * log_sx);
        float inv_s1_sq = expf(-2.0f * log_sy);
        float inv_s2_sq = expf(-2.0f * log_sz);

        // Precision = R * S^-2 * R^T
        // P[i][j] = sum_k(R[i][k] * (1/s_k^2) * R[j][k])

        // P[0][0]
        prec[0] = R[0] * R[0] * inv_s0_sq + R[3] * R[3] * inv_s1_sq + R[6] * R[6] * inv_s2_sq;
        // P[0][1] = P[1][0]
        prec[1] = R[0] * R[1] * inv_s0_sq + R[3] * R[4] * inv_s1_sq + R[6] * R[7] * inv_s2_sq;
        // P[0][2] = P[2][0]
        prec[2] = R[0] * R[2] * inv_s0_sq + R[3] * R[5] * inv_s1_sq + R[6] * R[8] * inv_s2_sq;
        // P[1][1]
        prec[3] = R[1] * R[1] * inv_s0_sq + R[4] * R[4] * inv_s1_sq + R[7] * R[7] * inv_s2_sq;
        // P[1][2] = P[2][1]
        prec[4] = R[1] * R[2] * inv_s0_sq + R[4] * R[5] * inv_s1_sq + R[7] * R[8] * inv_s2_sq;
        // P[2][2]
        prec[5] = R[2] * R[2] * inv_s0_sq + R[5] * R[5] * inv_s1_sq + R[8] * R[8] * inv_s2_sq;
    }

    /**
     * @brief Compute Mahalanobis distance squared: (x - mu)^T * Sigma^-1 * (x - mu)
     */
    __device__ __forceinline__ float mahalanobis_sq(
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
    __device__ __forceinline__ float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    /**
     * @brief Compute vacancy at a 3D point for a single Gaussian
     *
     * vacancy(x) = sqrt(max(0, 1 - o * exp(-0.5 * maha_sq)))
     */
    __device__ __forceinline__ float compute_vacancy_at_point(
        float px, float py, float pz,
        float mx, float my, float mz,
        const float* prec,
        float opacity) {

        float dx = px - mx;
        float dy = py - my;
        float dz = pz - mz;

        float maha_sq = mahalanobis_sq(dx, dy, dz, prec);
        float exponent = fminf(maha_sq * 0.5f, 88.0f);  // Clamp for numerical stability
        float occupancy = opacity * expf(-exponent);

        return sqrtf(fmaxf(0.0f, 1.0f - occupancy));
    }

    /**
     * @brief Compute peak depth t* for a Gaussian along a ray
     *
     * The peak depth is where the ray is closest to the Gaussian center,
     * which for an anisotropic Gaussian is where the Mahalanobis distance
     * is minimized along the ray.
     *
     * For ray r(t) = o + t*d, we minimize:
     *   ||Sigma^(-1/2) * (r(t) - mu)||^2
     *
     * Taking derivative and setting to zero:
     *   t* = (d^T * P * (mu - o)) / (d^T * P * d)
     *
     * where P = Sigma^-1 is the precision matrix.
     */
    __device__ __forceinline__ float compute_peak_depth(
        float ox, float oy, float oz,      // Ray origin
        float dx, float dy, float dz,      // Ray direction (normalized)
        float mx, float my, float mz,      // Gaussian mean
        const float* prec) {               // Precision matrix [6]

        // Compute (mu - o)
        float diff_x = mx - ox;
        float diff_y = my - oy;
        float diff_z = mz - oz;

        // Compute d^T * P * d (ray direction through precision)
        float dPd = prec[0] * dx * dx +
                    2.0f * prec[1] * dx * dy +
                    2.0f * prec[2] * dx * dz +
                    prec[3] * dy * dy +
                    2.0f * prec[4] * dy * dz +
                    prec[5] * dz * dz;

        // Compute d^T * P * (mu - o)
        float Pd_x = prec[0] * diff_x + prec[1] * diff_y + prec[2] * diff_z;
        float Pd_y = prec[1] * diff_x + prec[3] * diff_y + prec[4] * diff_z;
        float Pd_z = prec[2] * diff_x + prec[4] * diff_y + prec[5] * diff_z;

        float dPdiff = dx * Pd_x + dy * Pd_y + dz * Pd_z;

        // t* = (d^T * P * (mu - o)) / (d^T * P * d)
        float t_star = dPdiff / fmaxf(dPd, 1e-10f);

        return t_star;
    }

    /**
     * @brief Transform world point to camera coordinates
     */
    __device__ __forceinline__ void world_to_camera(
        float wx, float wy, float wz,
        const float* viewmat,  // [4, 4] column-major
        float& cx, float& cy, float& cz) {

        // viewmat is 4x4 column-major: [R|t; 0 0 0 1]
        // Column 0: R[:,0]
        // Column 1: R[:,1]
        // Column 2: R[:,2]
        // Column 3: t (translation)
        cx = viewmat[0] * wx + viewmat[4] * wy + viewmat[8] * wz + viewmat[12];
        cy = viewmat[1] * wx + viewmat[5] * wy + viewmat[9] * wz + viewmat[13];
        cz = viewmat[2] * wx + viewmat[6] * wy + viewmat[10] * wz + viewmat[14];
    }

    /**
     * @brief Create ray from pixel coordinates
     */
    __device__ __forceinline__ void pixel_to_ray(
        float px, float py,           // Pixel coordinates
        const float* viewmat,         // [4, 4] view matrix (world-to-camera)
        const float* K,               // [3, 3] intrinsics
        float& ox, float& oy, float& oz,    // Ray origin (world)
        float& dx, float& dy, float& dz) {  // Ray direction (world, normalized)

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

        // Extract camera-to-world rotation (transpose of world-to-camera rotation)
        // viewmat is [R|t], so R^T is the camera-to-world rotation
        // R^T[i,j] = R[j,i] = viewmat[i + j*4]
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
    // FORWARD KERNEL: Pre-computed vacancy arrays
    // =========================================================================

    /**
     * @brief Kernel for computing per-pixel transmittance with pre-computed vacancies
     *
     * Each thread handles one pixel. Uses tile-based iteration through Gaussians.
     */
    __global__ void compute_ray_transmittance_kernel(
        const float* __restrict__ depths,
        const float* __restrict__ t_stars,
        const float* __restrict__ vacancies_at_peak,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        float* __restrict__ transmittance_out,
        float query_t,
        int C,
        int H,
        int W,
        int M,
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

        // Return if out of bounds
        if (i >= H || j >= W) return;

        int pix_id = i * W + j;
        int global_pix_id = cid * H * W + pix_id;

        // Shift pointers for current camera
        tile_offsets += cid * tile_height * tile_width;
        depths += cid * M;
        t_stars += cid * M;
        vacancies_at_peak += cid * M;

        // Get range of Gaussians for this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];

        // Accumulate transmittance
        float T = 1.0f;

        for (int32_t idx = range_start; idx < range_end; ++idx) {
            int32_t g = flatten_ids[idx];

            float t_star = t_stars[g];
            float vacancy_peak = vacancies_at_peak[g];

            // For query_t, we need vacancy at query depth
            // In this simplified version, assume vacancy_at_t is provided or approximated
            // A more accurate version would compute vacancy at query_t on-the-fly

            // Simplified: use vacancy_at_peak as proxy for vacancy_at_t
            // This is exact when query_t == t_star
            float vacancy_at_t = vacancy_peak;  // Approximation

            float Ti = compute_stochastic_transmittance(
                query_t, t_star, vacancy_peak, vacancy_at_t);

            T *= Ti;

            // Early exit
            if (T < 1e-6f) {
                T = 0.0f;
                break;
            }
        }

        transmittance_out[global_pix_id] = T;
    }

    // =========================================================================
    // FORWARD KERNEL: Fused with on-the-fly vacancy computation
    // =========================================================================

    /**
     * @brief Fused kernel computing transmittance with vacancy on-the-fly
     *
     * More accurate but slower: computes vacancy at actual depths.
     */
    __global__ void compute_ray_transmittance_fused_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ viewmats,
        const float* __restrict__ Ks,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        float* __restrict__ transmittance_out,
        float query_t,
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

        // Shift pointers for current camera
        tile_offsets += cid * tile_height * tile_width;
        const float* viewmat = viewmats + cid * 16;
        const float* K = Ks + cid * 9;

        // Create ray for this pixel
        float ray_ox, ray_oy, ray_oz;
        float ray_dx, ray_dy, ray_dz;
        pixel_to_ray(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Get range of Gaussians for this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];

        // Accumulate transmittance
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

            float opacity = sigmoid(opacities[global_g]);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix(qw, qx, qy, qz, log_sx, log_sy, log_sz, prec);

            // Compute peak depth t* (closest approach to Gaussian center)
            float t_star = compute_peak_depth(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec);

            // Skip if t* is behind camera or too far
            if (t_star < 0.0f) continue;

            // Compute point on ray at t*
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;

            // Compute vacancy at t*
            float vacancy_at_t_star = compute_vacancy_at_point(
                pt_star_x, pt_star_y, pt_star_z,
                mx, my, mz,
                prec, opacity);

            // Compute point on ray at query_t
            float pt_t_x = ray_ox + query_t * ray_dx;
            float pt_t_y = ray_oy + query_t * ray_dy;
            float pt_t_z = ray_oz + query_t * ray_dz;

            // Compute vacancy at query_t
            float vacancy_at_t = compute_vacancy_at_point(
                pt_t_x, pt_t_y, pt_t_z,
                mx, my, mz,
                prec, opacity);

            // Compute stochastic transmittance for this Gaussian
            float Ti = compute_stochastic_transmittance(
                query_t, t_star, vacancy_at_t_star, vacancy_at_t);

            T *= Ti;

            // Early exit
            if (T < 1e-6f) {
                T = 0.0f;
                break;
            }
        }

        transmittance_out[global_pix_id] = T;
    }

    // =========================================================================
    // FORWARD KERNEL: Transmittance at rendered depth
    // =========================================================================

    /**
     * @brief Kernel for computing transmittance at per-pixel rendered depth
     */
    __global__ void compute_transmittance_at_rendered_depth_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ viewmats,
        const float* __restrict__ Ks,
        const float* __restrict__ rendered_depths,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        float* __restrict__ transmittance_out,
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

        // Get per-pixel query depth
        float query_t = rendered_depths[global_pix_id];

        // Handle invalid depths
        if (query_t <= 0.0f || !isfinite(query_t)) {
            transmittance_out[global_pix_id] = 1.0f;
            return;
        }

        // Shift pointers for current camera
        tile_offsets += cid * tile_height * tile_width;
        const float* viewmat = viewmats + cid * 16;
        const float* K = Ks + cid * 9;

        // Create ray for this pixel
        float ray_ox, ray_oy, ray_oz;
        float ray_dx, ray_dy, ray_dz;
        pixel_to_ray(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Get range of Gaussians for this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];

        // Accumulate transmittance
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

            float opacity = sigmoid(opacities[global_g]);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix(qw, qx, qy, qz, log_sx, log_sy, log_sz, prec);

            // Compute peak depth t*
            float t_star = compute_peak_depth(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec);

            // Skip if t* is behind camera
            if (t_star < 0.0f) continue;

            // Compute point on ray at t*
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;

            // Compute vacancy at t*
            float vacancy_at_t_star = compute_vacancy_at_point(
                pt_star_x, pt_star_y, pt_star_z,
                mx, my, mz,
                prec, opacity);

            // Compute point on ray at query_t
            float pt_t_x = ray_ox + query_t * ray_dx;
            float pt_t_y = ray_oy + query_t * ray_dy;
            float pt_t_z = ray_oz + query_t * ray_dz;

            // Compute vacancy at query_t
            float vacancy_at_t = compute_vacancy_at_point(
                pt_t_x, pt_t_y, pt_t_z,
                mx, my, mz,
                prec, opacity);

            // Compute stochastic transmittance
            float Ti = compute_stochastic_transmittance(
                query_t, t_star, vacancy_at_t_star, vacancy_at_t);

            T *= Ti;

            // Early exit
            if (T < 1e-6f) {
                T = 0.0f;
                break;
            }
        }

        transmittance_out[global_pix_id] = T;
    }

    // =========================================================================
    // BACKWARD KERNEL
    // =========================================================================

    /**
     * @brief Backward pass for stochastic transmittance
     *
     * Computes gradients w.r.t. Gaussian parameters using chain rule.
     */
    __global__ void compute_ray_transmittance_backward_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ viewmats,
        const float* __restrict__ Ks,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const int32_t* __restrict__ visible_indices,
        const float* __restrict__ grad_transmittance,
        float* __restrict__ grad_means,
        float* __restrict__ grad_quats,
        float* __restrict__ grad_scales,
        float* __restrict__ grad_opacities,
        float query_t,
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

        float grad_T = grad_transmittance[global_pix_id];

        // Skip if no gradient
        if (fabsf(grad_T) < 1e-10f) return;

        // Shift pointers for current camera
        tile_offsets += cid * tile_height * tile_width;
        const float* viewmat = viewmats + cid * 16;
        const float* K = Ks + cid * 9;

        // Create ray for this pixel
        float ray_ox, ray_oy, ray_oz;
        float ray_dx, ray_dy, ray_dz;
        pixel_to_ray(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Get range of Gaussians for this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];

        // First pass: compute T and per-Gaussian Ti values
        // (This is a simplified backward - full version would need recomputation)

        float T = 1.0f;

        for (int32_t idx = range_start; idx < range_end && T > 1e-6f; ++idx) {
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
            float opacity = sigmoid(raw_opacity);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix(qw, qx_q, qy_q, qz_q, log_sx, log_sy, log_sz, prec);

            // Compute peak depth t*
            float t_star = compute_peak_depth(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec);

            if (t_star < 0.0f) continue;

            // Compute vacancy at t* and query_t
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;

            float vacancy_at_t_star = compute_vacancy_at_point(
                pt_star_x, pt_star_y, pt_star_z,
                mx, my, mz,
                prec, opacity);

            float pt_t_x = ray_ox + query_t * ray_dx;
            float pt_t_y = ray_oy + query_t * ray_dy;
            float pt_t_z = ray_oz + query_t * ray_dz;

            float vacancy_at_t = compute_vacancy_at_point(
                pt_t_x, pt_t_y, pt_t_z,
                mx, my, mz,
                prec, opacity);

            float Ti = compute_stochastic_transmittance(
                query_t, t_star, vacancy_at_t_star, vacancy_at_t);

            // Gradient computation
            // dL/dTi = dL/dT * dT/dTi = grad_T * (T / Ti) for Ti > 0
            float grad_Ti = 0.0f;
            if (Ti > 1e-8f) {
                grad_Ti = grad_T * (T / Ti);
            }

            // dL/d(vacancy_at_t) and dL/d(vacancy_at_t_star) depend on branch
            float grad_v_t = 0.0f;
            float grad_v_t_star = 0.0f;

            if (query_t <= t_star) {
                // Ti = vacancy_at_t
                // dTi/d(vacancy_at_t) = 1
                grad_v_t = grad_Ti;
            } else {
                // Ti = vacancy_at_t_star^2 / vacancy_at_t
                // dTi/d(vacancy_at_t_star) = 2 * vacancy_at_t_star / vacancy_at_t
                // dTi/d(vacancy_at_t) = -vacancy_at_t_star^2 / vacancy_at_t^2
                if (vacancy_at_t > 1e-8f) {
                    grad_v_t_star = grad_Ti * 2.0f * vacancy_at_t_star / vacancy_at_t;
                    grad_v_t = grad_Ti * (-vacancy_at_t_star * vacancy_at_t_star) /
                               (vacancy_at_t * vacancy_at_t);
                }
            }

            // Backprop through vacancy computation to opacity
            // vacancy = sqrt(1 - o * G), where G = exp(-0.5 * maha)
            // d(vacancy)/d(opacity) = -G / (2 * vacancy) for vacancy > 0

            // Gradient w.r.t opacity (simplified - full version needs chain through precision)
            if (vacancy_at_t > 1e-8f) {
                float dx_t = pt_t_x - mx;
                float dy_t = pt_t_y - my;
                float dz_t = pt_t_z - mz;
                float maha_sq_t = mahalanobis_sq(dx_t, dy_t, dz_t, prec);
                float G_t = expf(-fminf(maha_sq_t * 0.5f, 88.0f));

                float dv_do = -G_t / (2.0f * vacancy_at_t);
                float grad_opacity = grad_v_t * dv_do;

                // Sigmoid backward: d(sigmoid)/d(raw) = sigmoid * (1 - sigmoid)
                float grad_raw_opacity = grad_opacity * opacity * (1.0f - opacity);

                atomicAdd(&grad_opacities[global_g], grad_raw_opacity);
            }

            if (vacancy_at_t_star > 1e-8f && fabsf(grad_v_t_star) > 1e-10f) {
                float dx_star = pt_star_x - mx;
                float dy_star = pt_star_y - my;
                float dz_star = pt_star_z - mz;
                float maha_sq_star = mahalanobis_sq(dx_star, dy_star, dz_star, prec);
                float G_star = expf(-fminf(maha_sq_star * 0.5f, 88.0f));

                float dv_do = -G_star / (2.0f * vacancy_at_t_star);
                float grad_opacity = grad_v_t_star * dv_do;
                float grad_raw_opacity = grad_opacity * opacity * (1.0f - opacity);

                atomicAdd(&grad_opacities[global_g], grad_raw_opacity);
            }

            // Note: Full gradient computation for means, quats, scales requires
            // backprop through precision matrix and Mahalanobis distance.
            // This is a simplified version focusing on opacity gradients.
            // Full implementation would add ~100 more lines for those gradients.

            T *= Ti;
        }
    }

    // =========================================================================
    // LAUNCH FUNCTIONS
    // =========================================================================

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        compute_ray_transmittance_kernel<<<grid, threads, 0, stream>>>(
            depths,
            t_stars,
            vacancies_at_peak,
            tile_offsets,
            flatten_ids,
            transmittance_out,
            query_t,
            C, H, W, M,
            tile_size,
            n_isects);
    }

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0 || N == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        compute_ray_transmittance_fused_kernel<<<grid, threads, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            tile_offsets,
            flatten_ids,
            visible_indices,
            transmittance_out,
            query_t,
            C, N, M, H, W,
            tile_size,
            n_isects);
    }

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0 || N == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        compute_ray_transmittance_backward_kernel<<<grid, threads, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            tile_offsets,
            flatten_ids,
            visible_indices,
            grad_transmittance,
            grad_means,
            grad_quats,
            grad_scales,
            grad_opacities,
            query_t,
            C, N, M, H, W,
            tile_size,
            n_isects);
    }

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0 || N == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        compute_transmittance_at_rendered_depth_kernel<<<grid, threads, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            rendered_depths,
            tile_offsets,
            flatten_ids,
            visible_indices,
            transmittance_out,
            C, N, M, H, W,
            tile_size,
            n_isects);
    }

} // namespace lfs::training::kernels
