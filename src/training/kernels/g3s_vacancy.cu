/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/g3s_vacancy.cuh"
#include <algorithm>

namespace lfs::training::kernels {

    // =============================================================================
    // DEVICE HELPER FUNCTIONS
    // =============================================================================

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
     * For efficiency, we compute the precision (inverse covariance) directly.
     * Since S is diagonal with scales on diagonal:
     *   S^-2 = diag(1/s0^2, 1/s1^2, 1/s2^2)
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
     *
     * Using the upper triangular precision matrix representation.
     */
    __device__ __forceinline__ float mahalanobis_sq(
        float dx, float dy, float dz,
        const float* prec) {

        // d^T * P * d where P is symmetric
        // = dx*(P00*dx + P01*dy + P02*dz) + dy*(P01*dx + P11*dy + P12*dz) + dz*(P02*dx + P12*dy + P22*dz)
        // = P00*dx*dx + 2*P01*dx*dy + 2*P02*dx*dz + P11*dy*dy + 2*P12*dy*dz + P22*dz*dz

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

    // =============================================================================
    // MAIN VACANCY KERNEL
    // =============================================================================

    /**
     * @brief Compute vacancy at query points
     *
     * Each thread handles one query point and iterates over all Gaussians.
     * This is the naive O(M*N) implementation - for large N, consider spatial
     * data structures.
     *
     * vacancy(x) = sqrt(max(0, 1 - G(x)))
     * where G(x) = sum_i(opacity_i * exp(-0.5 * mahalanobis_sq_i(x)))
     */
    __global__ void compute_vacancy_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ query_pts,
        float* __restrict__ vacancy_out,
        int N,
        int M) {

        int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m >= M) return;

        // Load query point
        float qx = query_pts[m * 3 + 0];
        float qy = query_pts[m * 3 + 1];
        float qz = query_pts[m * 3 + 2];

        // Accumulate Gaussian occupancy
        float occupancy = 0.0f;

        for (int n = 0; n < N; ++n) {
            // Load Gaussian parameters
            float mx = means[n * 3 + 0];
            float my = means[n * 3 + 1];
            float mz = means[n * 3 + 2];

            float qw = quats[n * 4 + 0];
            float quat_x = quats[n * 4 + 1];
            float quat_y = quats[n * 4 + 2];
            float quat_z = quats[n * 4 + 3];

            float log_sx = scales[n * 3 + 0];
            float log_sy = scales[n * 3 + 1];
            float log_sz = scales[n * 3 + 2];

            float opacity = sigmoid(opacities[n]);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix(qw, quat_x, quat_y, quat_z, log_sx, log_sy, log_sz, prec);

            // Compute displacement
            float dx = qx - mx;
            float dy = qy - my;
            float dz = qz - mz;

            // Compute Mahalanobis distance squared
            float maha_sq = mahalanobis_sq(dx, dy, dz, prec);

            // Gaussian contribution: o * exp(-0.5 * d^2)
            // Clamp exponent to avoid numerical issues
            float exponent = fminf(maha_sq * 0.5f, 88.0f); // exp(-88) ~= 0
            float contrib = opacity * expf(-exponent);

            occupancy += contrib;
        }

        // Vacancy = sqrt(1 - G(x)), clamped to [0, 1]
        float vacancy = sqrtf(fmaxf(0.0f, 1.0f - occupancy));
        vacancy_out[m] = vacancy;
    }

    // =============================================================================
    // CULLED VACANCY KERNEL (with distance threshold)
    // =============================================================================

    /**
     * @brief Compute vacancy with distance-based culling
     *
     * Skips Gaussians that are further than cull_radius from the query point.
     * This is a significant optimization for sparse scenes.
     */
    __global__ void compute_vacancy_culled_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ query_pts,
        float* __restrict__ vacancy_out,
        int N,
        int M,
        float cull_radius_sq) {

        int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m >= M) return;

        // Load query point
        float qx = query_pts[m * 3 + 0];
        float qy = query_pts[m * 3 + 1];
        float qz = query_pts[m * 3 + 2];

        // Accumulate Gaussian occupancy
        float occupancy = 0.0f;

        for (int n = 0; n < N; ++n) {
            // Load Gaussian center
            float mx = means[n * 3 + 0];
            float my = means[n * 3 + 1];
            float mz = means[n * 3 + 2];

            // Early culling based on Euclidean distance
            float dx = qx - mx;
            float dy = qy - my;
            float dz = qz - mz;
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > cull_radius_sq) continue;

            // Load remaining Gaussian parameters
            float qw = quats[n * 4 + 0];
            float quat_x = quats[n * 4 + 1];
            float quat_y = quats[n * 4 + 2];
            float quat_z = quats[n * 4 + 3];

            float log_sx = scales[n * 3 + 0];
            float log_sy = scales[n * 3 + 1];
            float log_sz = scales[n * 3 + 2];

            float opacity = sigmoid(opacities[n]);

            // Compute precision matrix
            float prec[6];
            compute_precision_matrix(qw, quat_x, quat_y, quat_z, log_sx, log_sy, log_sz, prec);

            // Compute Mahalanobis distance squared
            float maha_sq = mahalanobis_sq(dx, dy, dz, prec);

            // Gaussian contribution
            float exponent = fminf(maha_sq * 0.5f, 88.0f);
            float contrib = opacity * expf(-exponent);

            occupancy += contrib;
        }

        // Vacancy = sqrt(1 - G(x)), clamped to [0, 1]
        float vacancy = sqrtf(fmaxf(0.0f, 1.0f - occupancy));
        vacancy_out[m] = vacancy;
    }

    // =============================================================================
    // BACKWARD KERNEL
    // =============================================================================

    /**
     * @brief Backward pass for vacancy computation
     *
     * Computes gradient w.r.t. query points:
     *   dL/dx = dL/dv * dv/dG * dG/dx
     *
     * where:
     *   dv/dG = -1 / (2 * sqrt(1 - G)) = -1 / (2 * v)  (for v > 0)
     *   dG/dx = sum_i(o_i * exp(...) * (-Sigma_i^-1 * (x - x_c_i)))
     */
    __global__ void compute_vacancy_backward_kernel(
        const float* __restrict__ means,
        const float* __restrict__ quats,
        const float* __restrict__ scales,
        const float* __restrict__ opacities,
        const float* __restrict__ query_pts,
        const float* __restrict__ grad_vacancy,
        float* __restrict__ grad_query_pts,
        int N,
        int M) {

        int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m >= M) return;

        // Load query point
        float qx = query_pts[m * 3 + 0];
        float qy = query_pts[m * 3 + 1];
        float qz = query_pts[m * 3 + 2];

        // First pass: compute occupancy and vacancy for chain rule
        float occupancy = 0.0f;

        for (int n = 0; n < N; ++n) {
            float mx = means[n * 3 + 0];
            float my = means[n * 3 + 1];
            float mz = means[n * 3 + 2];

            float qw = quats[n * 4 + 0];
            float quat_x = quats[n * 4 + 1];
            float quat_y = quats[n * 4 + 2];
            float quat_z = quats[n * 4 + 3];

            float log_sx = scales[n * 3 + 0];
            float log_sy = scales[n * 3 + 1];
            float log_sz = scales[n * 3 + 2];

            float opacity = sigmoid(opacities[n]);

            float prec[6];
            compute_precision_matrix(qw, quat_x, quat_y, quat_z, log_sx, log_sy, log_sz, prec);

            float dx = qx - mx;
            float dy = qy - my;
            float dz = qz - mz;

            float maha_sq = mahalanobis_sq(dx, dy, dz, prec);
            float exponent = fminf(maha_sq * 0.5f, 88.0f);
            float contrib = opacity * expf(-exponent);

            occupancy += contrib;
        }

        float vacancy = sqrtf(fmaxf(1e-8f, 1.0f - occupancy));

        // dv/dG = -1 / (2 * v)
        float dv_dG = -0.5f / vacancy;

        // Get upstream gradient
        float grad_v = grad_vacancy[m];

        // dL/dG = dL/dv * dv/dG
        float dL_dG = grad_v * dv_dG;

        // Second pass: compute gradient w.r.t. query point
        float grad_x = 0.0f;
        float grad_y = 0.0f;
        float grad_z = 0.0f;

        for (int n = 0; n < N; ++n) {
            float mx = means[n * 3 + 0];
            float my = means[n * 3 + 1];
            float mz = means[n * 3 + 2];

            float qw = quats[n * 4 + 0];
            float quat_x = quats[n * 4 + 1];
            float quat_y = quats[n * 4 + 2];
            float quat_z = quats[n * 4 + 3];

            float log_sx = scales[n * 3 + 0];
            float log_sy = scales[n * 3 + 1];
            float log_sz = scales[n * 3 + 2];

            float opacity = sigmoid(opacities[n]);

            float prec[6];
            compute_precision_matrix(qw, quat_x, quat_y, quat_z, log_sx, log_sy, log_sz, prec);

            float dx = qx - mx;
            float dy = qy - my;
            float dz = qz - mz;

            float maha_sq = mahalanobis_sq(dx, dy, dz, prec);
            float exponent = fminf(maha_sq * 0.5f, 88.0f);
            float gauss_val = opacity * expf(-exponent);

            // dG_i/dx = gauss_val * (-Sigma^-1 * d)
            // For symmetric precision P, d(d^T P d)/dx = 2 * P * d
            // So dG_i/dx = -gauss_val * P * d

            // P * d
            float Pd_x = prec[0] * dx + prec[1] * dy + prec[2] * dz;
            float Pd_y = prec[1] * dx + prec[3] * dy + prec[4] * dz;
            float Pd_z = prec[2] * dx + prec[4] * dy + prec[5] * dz;

            // -gauss_val * P * d
            grad_x += -gauss_val * Pd_x;
            grad_y += -gauss_val * Pd_y;
            grad_z += -gauss_val * Pd_z;
        }

        // Apply chain rule: dL/dx = dL/dG * dG/dx
        grad_query_pts[m * 3 + 0] = dL_dG * grad_x;
        grad_query_pts[m * 3 + 1] = dL_dG * grad_y;
        grad_query_pts[m * 3 + 2] = dL_dG * grad_z;
    }

    // =============================================================================
    // LAUNCH FUNCTIONS
    // =============================================================================

    void launch_compute_vacancy(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* query_pts,
        float* vacancy_out,
        int N,
        int M,
        cudaStream_t stream) {

        if (M == 0 || N == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = (M + block_size - 1) / block_size;

        compute_vacancy_kernel<<<num_blocks, block_size, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            query_pts,
            vacancy_out,
            N,
            M);
    }

    void launch_compute_vacancy_culled(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* query_pts,
        float* vacancy_out,
        int N,
        int M,
        float cull_radius,
        cudaStream_t stream) {

        if (M == 0 || N == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = (M + block_size - 1) / block_size;
        float cull_radius_sq = cull_radius * cull_radius;

        compute_vacancy_culled_kernel<<<num_blocks, block_size, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            query_pts,
            vacancy_out,
            N,
            M,
            cull_radius_sq);
    }

    void launch_compute_vacancy_backward(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* query_pts,
        const float* grad_vacancy,
        float* grad_query_pts,
        int N,
        int M,
        cudaStream_t stream) {

        if (M == 0 || N == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = (M + block_size - 1) / block_size;

        compute_vacancy_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            query_pts,
            grad_vacancy,
            grad_query_pts,
            N,
            M);
    }

} // namespace lfs::training::kernels
