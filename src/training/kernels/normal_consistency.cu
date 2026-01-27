/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/normal_consistency.cuh"
#include <algorithm>

namespace lfs::training::kernels {

    // =============================================================================
    // DEVICE HELPER FUNCTIONS
    // =============================================================================

    /**
     * @brief Convert quaternion to 3x3 rotation matrix (column-major)
     *
     * Quaternion format: (w, x, y, z)
     * Returns column-major rotation matrix stored in 9 floats.
     * R[col * 3 + row] gives element at (row, col)
     */
    __device__ __forceinline__ void quat_to_rotmat_normal(
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
     * @brief Normalize a 3D vector
     */
    __device__ __forceinline__ void normalize3(float* v) {
        float inv_len = rsqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + 1e-12f);
        v[0] *= inv_len;
        v[1] *= inv_len;
        v[2] *= inv_len;
    }

    /**
     * @brief Dot product of two 3D vectors
     */
    __device__ __forceinline__ float dot3(const float* a, const float* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    /**
     * @brief Cross product of two 3D vectors: c = a x b
     */
    __device__ __forceinline__ void cross3(const float* a, const float* b, float* c) {
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
    }

    // =============================================================================
    // GAUSSIAN NORMAL COMPUTATION
    // =============================================================================

    /**
     * @brief Compute Gaussian normals from quaternions and scales
     *
     * The normal is the column of the rotation matrix corresponding to
     * the smallest scale (shortest ellipsoid axis).
     */
    __global__ void compute_gaussian_normals_kernel(
        const float* __restrict__ quaternions,
        const float* __restrict__ scales,
        float* __restrict__ normals_out,
        int N) {

        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N) return;

        // Load quaternion
        float qw = quaternions[n * 4 + 0];
        float qx = quaternions[n * 4 + 1];
        float qy = quaternions[n * 4 + 2];
        float qz = quaternions[n * 4 + 3];

        // Load scales (linear space, not log)
        float s0 = scales[n * 3 + 0];
        float s1 = scales[n * 3 + 1];
        float s2 = scales[n * 3 + 2];

        // Find the smallest scale (shortest axis = normal direction)
        int min_axis = 0;
        float min_scale = s0;
        if (s1 < min_scale) {
            min_axis = 1;
            min_scale = s1;
        }
        if (s2 < min_scale) {
            min_axis = 2;
        }

        // Convert quaternion to rotation matrix
        float R[9];
        quat_to_rotmat_normal(qw, qx, qy, qz, R);

        // Normal is the column of R corresponding to min_axis
        // Column min_axis: R[min_axis * 3 + 0], R[min_axis * 3 + 1], R[min_axis * 3 + 2]
        float nx = R[min_axis * 3 + 0];
        float ny = R[min_axis * 3 + 1];
        float nz = R[min_axis * 3 + 2];

        // Store normalized normal (already normalized from unit quaternion)
        normals_out[n * 3 + 0] = nx;
        normals_out[n * 3 + 1] = ny;
        normals_out[n * 3 + 2] = nz;
    }

    /**
     * @brief Backward pass for Gaussian normal computation
     *
     * Computes gradients w.r.t. quaternions and scales.
     * Note: Gradient w.r.t. scales is typically zero since the normal
     * direction only depends on which scale is smallest (discrete choice),
     * not the actual scale values.
     */
    __global__ void compute_gaussian_normals_backward_kernel(
        const float* __restrict__ quaternions,
        const float* __restrict__ scales,
        const float* __restrict__ grad_normals,
        float* __restrict__ grad_quaternions,
        float* __restrict__ grad_scales,
        int N) {

        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N) return;

        // Load quaternion
        float qw = quaternions[n * 4 + 0];
        float qx = quaternions[n * 4 + 1];
        float qy = quaternions[n * 4 + 2];
        float qz = quaternions[n * 4 + 3];

        // Load scales to find min axis
        float s0 = scales[n * 3 + 0];
        float s1 = scales[n * 3 + 1];
        float s2 = scales[n * 3 + 2];

        int min_axis = 0;
        float min_scale = s0;
        if (s1 < min_scale) {
            min_axis = 1;
            min_scale = s1;
        }
        if (s2 < min_scale) {
            min_axis = 2;
        }

        // Load gradient w.r.t. normal
        float gx = grad_normals[n * 3 + 0];
        float gy = grad_normals[n * 3 + 1];
        float gz = grad_normals[n * 3 + 2];

        // Normalize quaternion
        float qnorm_sq = qw * qw + qx * qx + qy * qy + qz * qz + 1e-12f;
        float inv_norm = rsqrtf(qnorm_sq);
        float w = qw * inv_norm;
        float x = qx * inv_norm;
        float y = qy * inv_norm;
        float z = qz * inv_norm;

        // Compute gradient of R[:, min_axis] w.r.t. normalized quaternion
        // R is computed from normalized quaternion (w, x, y, z)
        // Then we chain rule through normalization

        float dR_dw[3], dR_dx[3], dR_dy[3], dR_dz[3];

        if (min_axis == 0) {
            // Column 0: [1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy)]
            dR_dw[0] = 0.0f;
            dR_dw[1] = 2.0f * z;
            dR_dw[2] = -2.0f * y;

            dR_dx[0] = 0.0f;
            dR_dx[1] = 2.0f * y;
            dR_dx[2] = 2.0f * z;

            dR_dy[0] = -4.0f * y;
            dR_dy[1] = 2.0f * x;
            dR_dy[2] = -2.0f * w;

            dR_dz[0] = -4.0f * z;
            dR_dz[1] = 2.0f * w;
            dR_dz[2] = 2.0f * x;
        } else if (min_axis == 1) {
            // Column 1: [2(xy - wz), 1 - 2(x^2 + z^2), 2(yz + wx)]
            dR_dw[0] = -2.0f * z;
            dR_dw[1] = 0.0f;
            dR_dw[2] = 2.0f * x;

            dR_dx[0] = 2.0f * y;
            dR_dx[1] = -4.0f * x;
            dR_dx[2] = 2.0f * w;

            dR_dy[0] = 2.0f * x;
            dR_dy[1] = 0.0f;
            dR_dy[2] = 2.0f * z;

            dR_dz[0] = -2.0f * w;
            dR_dz[1] = -4.0f * z;
            dR_dz[2] = 2.0f * y;
        } else {
            // Column 2: [2(xz + wy), 2(yz - wx), 1 - 2(x^2 + y^2)]
            dR_dw[0] = 2.0f * y;
            dR_dw[1] = -2.0f * x;
            dR_dw[2] = 0.0f;

            dR_dx[0] = 2.0f * z;
            dR_dx[1] = -2.0f * w;
            dR_dx[2] = -4.0f * x;

            dR_dy[0] = 2.0f * w;
            dR_dy[1] = 2.0f * z;
            dR_dy[2] = -4.0f * y;

            dR_dz[0] = 2.0f * x;
            dR_dz[1] = 2.0f * y;
            dR_dz[2] = 0.0f;
        }

        // Gradient w.r.t. normalized quaternion
        float g_w = gx * dR_dw[0] + gy * dR_dw[1] + gz * dR_dw[2];
        float g_x = gx * dR_dx[0] + gy * dR_dx[1] + gz * dR_dx[2];
        float g_y = gx * dR_dy[0] + gy * dR_dy[1] + gz * dR_dy[2];
        float g_z = gx * dR_dz[0] + gy * dR_dz[1] + gz * dR_dz[2];

        // Chain rule through normalization
        // d(q_normalized) / d(q_raw) = (I - q_normalized * q_normalized^T) / ||q||
        float qn_dot_g = w * g_w + x * g_x + y * g_y + z * g_z;

        float grad_qw = (g_w - w * qn_dot_g) * inv_norm;
        float grad_qx = (g_x - x * qn_dot_g) * inv_norm;
        float grad_qy = (g_y - y * qn_dot_g) * inv_norm;
        float grad_qz = (g_z - z * qn_dot_g) * inv_norm;

        // Accumulate gradients
        atomicAdd(&grad_quaternions[n * 4 + 0], grad_qw);
        atomicAdd(&grad_quaternions[n * 4 + 1], grad_qx);
        atomicAdd(&grad_quaternions[n * 4 + 2], grad_qy);
        atomicAdd(&grad_quaternions[n * 4 + 3], grad_qz);

        // Gradient w.r.t. scales is zero for normal direction
        // (min_axis is a discrete choice, not differentiable)
        // However, we could add a soft version if needed in the future
    }

    // =============================================================================
    // DEPTH NORMAL COMPUTATION
    // =============================================================================

    /**
     * @brief Compute surface normals from depth map using Sobel-like finite differences
     *
     * For a pixel at (u, v) with depth z:
     * - World X displacement per pixel: dx_world = z / fx
     * - World Y displacement per pixel: dy_world = z / fy
     *
     * Surface normal = normalize(dP/du x dP/dv)
     * where dP/du = (z/fx, 0, dz/du) and dP/dv = (0, z/fy, dz/dv)
     *
     * Cross product gives: n = (-z/fx * dz/dv, -z/fy * dz/du, z^2/(fx*fy))
     * After normalization, this simplifies significantly.
     */
    __global__ void compute_depth_normals_kernel(
        const float* __restrict__ depth_map,
        float* __restrict__ normals_out,
        int H,
        int W,
        float fx,
        float fy) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= W || y >= H) return;

        int idx = y * W + x;

        // Boundary check - set invalid normals to zero
        if (x == 0 || x >= W - 1 || y == 0 || y >= H - 1) {
            normals_out[idx * 3 + 0] = 0.0f;
            normals_out[idx * 3 + 1] = 0.0f;
            normals_out[idx * 3 + 2] = 0.0f;
            return;
        }

        float z = depth_map[idx];

        // Skip invalid depth
        if (z <= 0.0f || !isfinite(z)) {
            normals_out[idx * 3 + 0] = 0.0f;
            normals_out[idx * 3 + 1] = 0.0f;
            normals_out[idx * 3 + 2] = 0.0f;
            return;
        }

        // Load neighboring depths for Sobel gradient
        float z_left = depth_map[y * W + (x - 1)];
        float z_right = depth_map[y * W + (x + 1)];
        float z_up = depth_map[(y - 1) * W + x];
        float z_down = depth_map[(y + 1) * W + x];

        // Check if neighbors are valid
        if (z_left <= 0.0f || z_right <= 0.0f || z_up <= 0.0f || z_down <= 0.0f ||
            !isfinite(z_left) || !isfinite(z_right) || !isfinite(z_up) || !isfinite(z_down)) {
            normals_out[idx * 3 + 0] = 0.0f;
            normals_out[idx * 3 + 1] = 0.0f;
            normals_out[idx * 3 + 2] = 0.0f;
            return;
        }

        // Simple central difference for depth gradients
        float dz_dx = (z_right - z_left) * 0.5f;
        float dz_dy = (z_down - z_up) * 0.5f;

        // Tangent vectors in world space:
        // dP/dx = (z/fx, 0, dz_dx)  - moving one pixel right
        // dP/dy = (0, z/fy, dz_dy)  - moving one pixel down
        //
        // Normal = dP/dx x dP/dy
        //   = (0 * dz_dy - dz_dx * z/fy,
        //      dz_dx * 0 - z/fx * dz_dy,
        //      z/fx * z/fy - 0 * 0)
        //   = (-dz_dx * z/fy, -z/fx * dz_dy, z^2/(fx*fy))

        // Simplify by multiplying by fx * fy
        float nx = -dz_dx * fx;
        float ny = -dz_dy * fy;
        float nz = z;

        // Normalize
        float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
        nx *= inv_len;
        ny *= inv_len;
        nz *= inv_len;

        // Store normal (pointing towards camera, i.e., positive z)
        // Flip sign if nz < 0 to ensure consistent orientation
        if (nz < 0.0f) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }

        normals_out[idx * 3 + 0] = nx;
        normals_out[idx * 3 + 1] = ny;
        normals_out[idx * 3 + 2] = nz;
    }

    /**
     * @brief Backward pass for depth normal computation
     *
     * Computes gradient w.r.t. depth given gradient w.r.t. normals.
     */
    __global__ void compute_depth_normals_backward_kernel(
        const float* __restrict__ depth_map,
        const float* __restrict__ normals,
        const float* __restrict__ grad_normals,
        float* __restrict__ grad_depth,
        int H,
        int W,
        float fx,
        float fy) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= W || y >= H) return;

        int idx = y * W + x;

        // Skip boundary pixels
        if (x == 0 || x >= W - 1 || y == 0 || y >= H - 1) {
            return;
        }

        float z = depth_map[idx];
        if (z <= 0.0f || !isfinite(z)) {
            return;
        }

        // Check if this pixel has valid normal
        float nz = normals[idx * 3 + 2];
        if (nz == 0.0f) {
            return;
        }

        // Load gradient w.r.t. normal
        float gx = grad_normals[idx * 3 + 0];
        float gy = grad_normals[idx * 3 + 1];
        float gz = grad_normals[idx * 3 + 2];

        // Recompute intermediate values
        float z_left = depth_map[y * W + (x - 1)];
        float z_right = depth_map[y * W + (x + 1)];
        float z_up = depth_map[(y - 1) * W + x];
        float z_down = depth_map[(y + 1) * W + x];

        if (z_left <= 0.0f || z_right <= 0.0f || z_up <= 0.0f || z_down <= 0.0f) {
            return;
        }

        float dz_dx = (z_right - z_left) * 0.5f;
        float dz_dy = (z_down - z_up) * 0.5f;

        // Unnormalized normal: (-dz_dx * fx, -dz_dy * fy, z)
        float ux = -dz_dx * fx;
        float uy = -dz_dy * fy;
        float uz = z;

        float norm_sq = ux * ux + uy * uy + uz * uz + 1e-12f;
        float norm = sqrtf(norm_sq);
        float inv_norm = 1.0f / norm;
        float inv_norm3 = inv_norm * inv_norm * inv_norm;

        // Gradient of normalize operation:
        // d(u/||u||)/du = (I - n*n^T) / ||u||
        // where n = u/||u||

        float nx = ux * inv_norm;
        float ny = uy * inv_norm;
        float nz_comp = uz * inv_norm;

        // d(normalized)/d(unnormalized) applied to grad
        float grad_ux = inv_norm * (gx - nx * (nx * gx + ny * gy + nz_comp * gz));
        float grad_uy = inv_norm * (gy - ny * (nx * gx + ny * gy + nz_comp * gz));
        float grad_uz = inv_norm * (gz - nz_comp * (nx * gx + ny * gy + nz_comp * gz));

        // grad_uz contributes to z
        atomicAdd(&grad_depth[idx], grad_uz);

        // grad_ux = -fx * d(dz_dx)/d(depth)
        // dz_dx = (z_right - z_left) * 0.5
        // d(dz_dx)/d(z_left) = -0.5
        // d(dz_dx)/d(z_right) = 0.5
        atomicAdd(&grad_depth[y * W + (x - 1)], grad_ux * fx * 0.5f);
        atomicAdd(&grad_depth[y * W + (x + 1)], -grad_ux * fx * 0.5f);

        // grad_uy = -fy * d(dz_dy)/d(depth)
        // dz_dy = (z_down - z_up) * 0.5
        atomicAdd(&grad_depth[(y - 1) * W + x], grad_uy * fy * 0.5f);
        atomicAdd(&grad_depth[(y + 1) * W + x], -grad_uy * fy * 0.5f);
    }

    // =============================================================================
    // NORMAL CONSISTENCY FORWARD
    // =============================================================================

    /**
     * @brief Compute normal consistency loss per pixel
     *
     * loss_pixel = sum_k(weight_k * (1 - gaussian_normal_k dot depth_normal))
     */
    __global__ void normal_consistency_forward_kernel(
        const float* __restrict__ gaussian_normals,
        const float* __restrict__ depth_normals,
        const float* __restrict__ blend_weights,
        const int* __restrict__ gaussian_indices,
        float* __restrict__ loss_per_pixel,
        int H,
        int W,
        int K) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= W || y >= H) return;

        int pixel_idx = y * W + x;

        // Load depth normal for this pixel
        float dn_x = depth_normals[pixel_idx * 3 + 0];
        float dn_y = depth_normals[pixel_idx * 3 + 1];
        float dn_z = depth_normals[pixel_idx * 3 + 2];

        // Check if depth normal is valid
        float dn_len_sq = dn_x * dn_x + dn_y * dn_y + dn_z * dn_z;
        if (dn_len_sq < 0.5f) {
            // Invalid depth normal
            loss_per_pixel[pixel_idx] = 0.0f;
            return;
        }

        float loss = 0.0f;

        // Iterate over contributing Gaussians
        for (int k = 0; k < K; ++k) {
            int gauss_idx = gaussian_indices[pixel_idx * K + k];
            if (gauss_idx < 0) break; // No more Gaussians for this pixel

            float weight = blend_weights[pixel_idx * K + k];
            if (weight <= 0.0f) continue;

            // Load Gaussian normal
            float gn_x = gaussian_normals[gauss_idx * 3 + 0];
            float gn_y = gaussian_normals[gauss_idx * 3 + 1];
            float gn_z = gaussian_normals[gauss_idx * 3 + 2];

            // Dot product
            float dot = gn_x * dn_x + gn_y * dn_y + gn_z * dn_z;

            // Use absolute value of dot product since normals can be flipped
            // Loss = weight * (1 - |dot|)
            loss += weight * (1.0f - fabsf(dot));
        }

        loss_per_pixel[pixel_idx] = loss;
    }

    /**
     * @brief Backward pass for normal consistency
     *
     * Computes gradient w.r.t. Gaussian normals.
     */
    __global__ void normal_consistency_backward_kernel(
        const float* __restrict__ gaussian_normals,
        const float* __restrict__ depth_normals,
        const float* __restrict__ blend_weights,
        const int* __restrict__ gaussian_indices,
        const float* __restrict__ grad_loss_per_pixel,
        float* __restrict__ grad_gaussian_normals,
        int N,
        int H,
        int W,
        int K) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= W || y >= H) return;

        int pixel_idx = y * W + x;

        // Load upstream gradient
        float grad_loss = grad_loss_per_pixel[pixel_idx];
        if (grad_loss == 0.0f) return;

        // Load depth normal
        float dn_x = depth_normals[pixel_idx * 3 + 0];
        float dn_y = depth_normals[pixel_idx * 3 + 1];
        float dn_z = depth_normals[pixel_idx * 3 + 2];

        float dn_len_sq = dn_x * dn_x + dn_y * dn_y + dn_z * dn_z;
        if (dn_len_sq < 0.5f) return;

        // Compute gradients for each Gaussian
        for (int k = 0; k < K; ++k) {
            int gauss_idx = gaussian_indices[pixel_idx * K + k];
            if (gauss_idx < 0) break;

            float weight = blend_weights[pixel_idx * K + k];
            if (weight <= 0.0f) continue;

            // Load Gaussian normal
            float gn_x = gaussian_normals[gauss_idx * 3 + 0];
            float gn_y = gaussian_normals[gauss_idx * 3 + 1];
            float gn_z = gaussian_normals[gauss_idx * 3 + 2];

            float dot = gn_x * dn_x + gn_y * dn_y + gn_z * dn_z;

            // d(1 - |dot|)/d(dot) = -sign(dot)
            float sign_dot = (dot >= 0.0f) ? 1.0f : -1.0f;

            // Gradient contribution: grad_loss * weight * (-sign(dot)) * depth_normal
            float coeff = grad_loss * weight * (-sign_dot);

            atomicAdd(&grad_gaussian_normals[gauss_idx * 3 + 0], coeff * dn_x);
            atomicAdd(&grad_gaussian_normals[gauss_idx * 3 + 1], coeff * dn_y);
            atomicAdd(&grad_gaussian_normals[gauss_idx * 3 + 2], coeff * dn_z);
        }
    }

    // =============================================================================
    // FUSED FORWARD + BACKWARD
    // =============================================================================

    /**
     * @brief Fused normal consistency: compute loss and gradients in one pass
     *
     * More memory-efficient for training as it avoids storing intermediate
     * Gaussian normal gradients.
     */
    __global__ void fused_normal_consistency_kernel(
        const float* __restrict__ quaternions,
        const float* __restrict__ scales,
        const float* __restrict__ depth_normals,
        const float* __restrict__ blend_weights,
        const int* __restrict__ gaussian_indices,
        float* __restrict__ loss_out,
        float* __restrict__ grad_quaternions,
        float* __restrict__ grad_scales,
        int N,
        int H,
        int W,
        int K,
        float weight) {

        // Use shared memory for block-level loss reduction
        __shared__ float block_loss;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            block_loss = 0.0f;
        }
        __syncthreads();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float local_loss = 0.0f;

        if (x < W && y < H) {
            int pixel_idx = y * W + x;

            // Load depth normal
            float dn_x = depth_normals[pixel_idx * 3 + 0];
            float dn_y = depth_normals[pixel_idx * 3 + 1];
            float dn_z = depth_normals[pixel_idx * 3 + 2];

            float dn_len_sq = dn_x * dn_x + dn_y * dn_y + dn_z * dn_z;

            if (dn_len_sq >= 0.5f) {
                // Process each contributing Gaussian
                for (int k = 0; k < K; ++k) {
                    int gauss_idx = gaussian_indices[pixel_idx * K + k];
                    if (gauss_idx < 0) break;

                    float blend_weight = blend_weights[pixel_idx * K + k];
                    if (blend_weight <= 0.0f) continue;

                    // Load quaternion and scales for this Gaussian
                    float qw = quaternions[gauss_idx * 4 + 0];
                    float qx = quaternions[gauss_idx * 4 + 1];
                    float qy = quaternions[gauss_idx * 4 + 2];
                    float qz = quaternions[gauss_idx * 4 + 3];

                    float s0 = expf(scales[gauss_idx * 3 + 0]);
                    float s1 = expf(scales[gauss_idx * 3 + 1]);
                    float s2 = expf(scales[gauss_idx * 3 + 2]);

                    // Find min axis
                    int min_axis = 0;
                    float min_scale = s0;
                    if (s1 < min_scale) {
                        min_axis = 1;
                        min_scale = s1;
                    }
                    if (s2 < min_scale) {
                        min_axis = 2;
                    }

                    // Compute Gaussian normal (from rotation matrix column)
                    float R[9];
                    quat_to_rotmat_normal(qw, qx, qy, qz, R);

                    float gn_x = R[min_axis * 3 + 0];
                    float gn_y = R[min_axis * 3 + 1];
                    float gn_z = R[min_axis * 3 + 2];

                    // Compute dot product and loss
                    float dot = gn_x * dn_x + gn_y * dn_y + gn_z * dn_z;
                    float abs_dot = fabsf(dot);
                    float pixel_loss = blend_weight * (1.0f - abs_dot);
                    local_loss += pixel_loss;

                    // Backward pass
                    float sign_dot = (dot >= 0.0f) ? 1.0f : -1.0f;
                    float grad_coeff = weight * blend_weight * (-sign_dot);

                    // Gradient w.r.t. Gaussian normal
                    float gn_grad_x = grad_coeff * dn_x;
                    float gn_grad_y = grad_coeff * dn_y;
                    float gn_grad_z = grad_coeff * dn_z;

                    // Backprop through rotation matrix to quaternion
                    // (Similar to backward kernel above)
                    float qnorm_sq = qw * qw + qx * qx + qy * qy + qz * qz + 1e-12f;
                    float inv_norm = rsqrtf(qnorm_sq);
                    float w = qw * inv_norm;
                    float x_n = qx * inv_norm;
                    float y_n = qy * inv_norm;
                    float z_n = qz * inv_norm;

                    float dR_dw[3], dR_dx[3], dR_dy[3], dR_dz[3];

                    if (min_axis == 0) {
                        dR_dw[0] = 0.0f; dR_dw[1] = 2.0f * z_n; dR_dw[2] = -2.0f * y_n;
                        dR_dx[0] = 0.0f; dR_dx[1] = 2.0f * y_n; dR_dx[2] = 2.0f * z_n;
                        dR_dy[0] = -4.0f * y_n; dR_dy[1] = 2.0f * x_n; dR_dy[2] = -2.0f * w;
                        dR_dz[0] = -4.0f * z_n; dR_dz[1] = 2.0f * w; dR_dz[2] = 2.0f * x_n;
                    } else if (min_axis == 1) {
                        dR_dw[0] = -2.0f * z_n; dR_dw[1] = 0.0f; dR_dw[2] = 2.0f * x_n;
                        dR_dx[0] = 2.0f * y_n; dR_dx[1] = -4.0f * x_n; dR_dx[2] = 2.0f * w;
                        dR_dy[0] = 2.0f * x_n; dR_dy[1] = 0.0f; dR_dy[2] = 2.0f * z_n;
                        dR_dz[0] = -2.0f * w; dR_dz[1] = -4.0f * z_n; dR_dz[2] = 2.0f * y_n;
                    } else {
                        dR_dw[0] = 2.0f * y_n; dR_dw[1] = -2.0f * x_n; dR_dw[2] = 0.0f;
                        dR_dx[0] = 2.0f * z_n; dR_dx[1] = -2.0f * w; dR_dx[2] = -4.0f * x_n;
                        dR_dy[0] = 2.0f * w; dR_dy[1] = 2.0f * z_n; dR_dy[2] = -4.0f * y_n;
                        dR_dz[0] = 2.0f * x_n; dR_dz[1] = 2.0f * y_n; dR_dz[2] = 0.0f;
                    }

                    float g_w = gn_grad_x * dR_dw[0] + gn_grad_y * dR_dw[1] + gn_grad_z * dR_dw[2];
                    float g_x = gn_grad_x * dR_dx[0] + gn_grad_y * dR_dx[1] + gn_grad_z * dR_dx[2];
                    float g_y = gn_grad_x * dR_dy[0] + gn_grad_y * dR_dy[1] + gn_grad_z * dR_dy[2];
                    float g_z = gn_grad_x * dR_dz[0] + gn_grad_y * dR_dz[1] + gn_grad_z * dR_dz[2];

                    float qn_dot_g = w * g_w + x_n * g_x + y_n * g_y + z_n * g_z;

                    atomicAdd(&grad_quaternions[gauss_idx * 4 + 0], (g_w - w * qn_dot_g) * inv_norm);
                    atomicAdd(&grad_quaternions[gauss_idx * 4 + 1], (g_x - x_n * qn_dot_g) * inv_norm);
                    atomicAdd(&grad_quaternions[gauss_idx * 4 + 2], (g_y - y_n * qn_dot_g) * inv_norm);
                    atomicAdd(&grad_quaternions[gauss_idx * 4 + 3], (g_z - z_n * qn_dot_g) * inv_norm);
                }
            }
        }

        // Reduce loss within block
        local_loss = lfs::core::warp_ops::block_reduce_sum(local_loss);

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            atomicAdd(loss_out, weight * local_loss);
        }
    }

    // =============================================================================
    // LAUNCH FUNCTIONS
    // =============================================================================

    void launch_compute_gaussian_normals(
        const float* quaternions,
        const float* scales,
        float* normals_out,
        int N,
        cudaStream_t stream) {

        if (N == 0) return;

        const int block_size = 256;
        const int num_blocks = (N + block_size - 1) / block_size;

        compute_gaussian_normals_kernel<<<num_blocks, block_size, 0, stream>>>(
            quaternions, scales, normals_out, N);
    }

    void launch_compute_gaussian_normals_backward(
        const float* quaternions,
        const float* scales,
        const float* grad_normals,
        float* grad_quaternions,
        float* grad_scales,
        int N,
        cudaStream_t stream) {

        if (N == 0) return;

        const int block_size = 256;
        const int num_blocks = (N + block_size - 1) / block_size;

        compute_gaussian_normals_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
            quaternions, scales, grad_normals, grad_quaternions, grad_scales, N);
    }

    void launch_compute_depth_normals(
        const float* depth_map,
        float* normals_out,
        int H,
        int W,
        float fx,
        float fy,
        cudaStream_t stream) {

        if (H == 0 || W == 0) return;

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        compute_depth_normals_kernel<<<grid, block, 0, stream>>>(
            depth_map, normals_out, H, W, fx, fy);
    }

    void launch_compute_depth_normals_backward(
        const float* depth_map,
        const float* normals,
        const float* grad_normals,
        float* grad_depth,
        int H,
        int W,
        float fx,
        float fy,
        cudaStream_t stream) {

        if (H == 0 || W == 0) return;

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        compute_depth_normals_backward_kernel<<<grid, block, 0, stream>>>(
            depth_map, normals, grad_normals, grad_depth, H, W, fx, fy);
    }

    void launch_normal_consistency_forward(
        const float* gaussian_normals,
        const float* depth_normals,
        const float* blend_weights,
        const int* gaussian_indices,
        float* loss_per_pixel,
        int H,
        int W,
        int K,
        cudaStream_t stream) {

        if (H == 0 || W == 0) return;

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        normal_consistency_forward_kernel<<<grid, block, 0, stream>>>(
            gaussian_normals, depth_normals, blend_weights, gaussian_indices,
            loss_per_pixel, H, W, K);
    }

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
        cudaStream_t stream) {

        if (H == 0 || W == 0 || N == 0) return;

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        normal_consistency_backward_kernel<<<grid, block, 0, stream>>>(
            gaussian_normals, depth_normals, blend_weights, gaussian_indices,
            grad_loss_per_pixel, grad_gaussian_normals, N, H, W, K);
    }

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
        cudaStream_t stream) {

        if (H == 0 || W == 0 || N == 0 || weight == 0.0f) return;

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

        fused_normal_consistency_kernel<<<grid, block, 0, stream>>>(
            quaternions, scales, depth_normals, blend_weights, gaussian_indices,
            loss_out, grad_quaternions, grad_scales, N, H, W, K, weight);
    }

} // namespace lfs::training::kernels
