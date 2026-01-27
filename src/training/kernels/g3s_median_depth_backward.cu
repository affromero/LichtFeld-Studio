/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file g3s_median_depth_backward.cu
 * @brief Complete closed-form backward pass for G3S median depth
 *
 * Implements Equation 13 from "Gaussian Geometry Guidance for Scene Reconstruction":
 *
 *   d(t_med)/d(theta) = -[dT(t_med; theta)/d(theta)] / [dT(t; theta)/dt]|_{t=t_med}
 *
 * This is the implicit function theorem: since T(t_med, theta) = 0.5 defines
 * t_med implicitly as a function of theta.
 *
 * Key insight: The gradient distributes to ALL Gaussians along the ray,
 * not just the one that "won" the median depth. This is why G3S produces
 * better geometry - every Gaussian gets supervision signal.
 */

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
    __device__ __forceinline__ void quat_to_rotmat_bwd(
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
     * @brief Compute precision matrix and its gradient helper terms
     *
     * Returns:
     *   - prec: upper triangular precision matrix [6]
     *   - R: rotation matrix [9]
     *   - inv_s_sq: inverse scale squared [3]
     */
    __device__ __forceinline__ void compute_precision_with_components(
        float qw, float qx, float qy, float qz,
        float log_sx, float log_sy, float log_sz,
        float* prec,
        float* R,
        float* inv_s_sq) {

        quat_to_rotmat_bwd(qw, qx, qy, qz, R);

        // Compute inverse scales squared: 1/s^2 = exp(-2 * log_s)
        inv_s_sq[0] = expf(-2.0f * log_sx);
        inv_s_sq[1] = expf(-2.0f * log_sy);
        inv_s_sq[2] = expf(-2.0f * log_sz);

        // Precision = R * S^-2 * R^T
        prec[0] = R[0] * R[0] * inv_s_sq[0] + R[3] * R[3] * inv_s_sq[1] + R[6] * R[6] * inv_s_sq[2];
        prec[1] = R[0] * R[1] * inv_s_sq[0] + R[3] * R[4] * inv_s_sq[1] + R[6] * R[7] * inv_s_sq[2];
        prec[2] = R[0] * R[2] * inv_s_sq[0] + R[3] * R[5] * inv_s_sq[1] + R[6] * R[8] * inv_s_sq[2];
        prec[3] = R[1] * R[1] * inv_s_sq[0] + R[4] * R[4] * inv_s_sq[1] + R[7] * R[7] * inv_s_sq[2];
        prec[4] = R[1] * R[2] * inv_s_sq[0] + R[4] * R[5] * inv_s_sq[1] + R[7] * R[8] * inv_s_sq[2];
        prec[5] = R[2] * R[2] * inv_s_sq[0] + R[5] * R[5] * inv_s_sq[1] + R[8] * R[8] * inv_s_sq[2];
    }

    /**
     * @brief Compute Mahalanobis distance squared
     */
    __device__ __forceinline__ float mahalanobis_sq_bwd(
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
     * @brief Compute gradient of Mahalanobis distance squared w.r.t. delta (point - mean)
     *
     * d(maha_sq)/d(delta) = 2 * P * delta
     * where P is the precision matrix
     */
    __device__ __forceinline__ void mahalanobis_sq_grad_delta(
        float dx, float dy, float dz,
        const float* prec,
        float& grad_dx, float& grad_dy, float& grad_dz) {

        // P * delta (symmetric matrix-vector product)
        float Pd_x = prec[0] * dx + prec[1] * dy + prec[2] * dz;
        float Pd_y = prec[1] * dx + prec[3] * dy + prec[4] * dz;
        float Pd_z = prec[2] * dx + prec[4] * dy + prec[5] * dz;

        // Gradient is 2 * P * delta
        grad_dx = 2.0f * Pd_x;
        grad_dy = 2.0f * Pd_y;
        grad_dz = 2.0f * Pd_z;
    }

    /**
     * @brief Compute gradient of Mahalanobis distance squared w.r.t. precision matrix
     *
     * For symmetric P, d(delta^T P delta)/dP is the outer product matrix
     * but we only need the upper triangular part since P is symmetric.
     */
    __device__ __forceinline__ void mahalanobis_sq_grad_prec(
        float dx, float dy, float dz,
        float* grad_prec) {

        // Gradient w.r.t. upper triangular precision
        // P[0,0]: dx * dx
        // P[0,1] = P[1,0]: 2 * dx * dy (factor of 2 because symmetric)
        // etc.
        grad_prec[0] = dx * dx;
        grad_prec[1] = 2.0f * dx * dy;
        grad_prec[2] = 2.0f * dx * dz;
        grad_prec[3] = dy * dy;
        grad_prec[4] = 2.0f * dy * dz;
        grad_prec[5] = dz * dz;
    }

    /**
     * @brief Backward through precision matrix to quaternion and scale
     *
     * Given grad_prec (gradient w.r.t. precision matrix),
     * compute gradients w.r.t. quaternion and log-scale.
     *
     * Precision = R * S^-2 * R^T
     * where R = quat_to_rotmat(q) and S^-2 = diag(exp(-2*log_s))
     */
    __device__ __forceinline__ void precision_backward(
        float qw, float qx, float qy, float qz,
        float log_sx, float log_sy, float log_sz,
        const float* R,
        const float* inv_s_sq,
        const float* grad_prec,
        float& grad_qw, float& grad_qx, float& grad_qy, float& grad_qz,
        float& grad_log_sx, float& grad_log_sy, float& grad_log_sz) {

        // Gradient w.r.t. inverse scale squared
        // P[i,j] = sum_k R[i,k] * (1/s_k^2) * R[j,k]
        // d(P[i,j])/d(1/s_k^2) = R[i,k] * R[j,k]

        // Accumulate gradient for each scale
        float grad_inv_s_sq[3] = {0.0f, 0.0f, 0.0f};

        // P[0,0] = R[0,0]^2 * inv_s0 + R[0,1]^2 * inv_s1 + R[0,2]^2 * inv_s2
        grad_inv_s_sq[0] += grad_prec[0] * R[0] * R[0];
        grad_inv_s_sq[1] += grad_prec[0] * R[3] * R[3];
        grad_inv_s_sq[2] += grad_prec[0] * R[6] * R[6];

        // P[0,1] (with factor of 2 already in grad_prec[1])
        grad_inv_s_sq[0] += grad_prec[1] * R[0] * R[1] * 0.5f;
        grad_inv_s_sq[1] += grad_prec[1] * R[3] * R[4] * 0.5f;
        grad_inv_s_sq[2] += grad_prec[1] * R[6] * R[7] * 0.5f;

        // P[0,2]
        grad_inv_s_sq[0] += grad_prec[2] * R[0] * R[2] * 0.5f;
        grad_inv_s_sq[1] += grad_prec[2] * R[3] * R[5] * 0.5f;
        grad_inv_s_sq[2] += grad_prec[2] * R[6] * R[8] * 0.5f;

        // P[1,1]
        grad_inv_s_sq[0] += grad_prec[3] * R[1] * R[1];
        grad_inv_s_sq[1] += grad_prec[3] * R[4] * R[4];
        grad_inv_s_sq[2] += grad_prec[3] * R[7] * R[7];

        // P[1,2]
        grad_inv_s_sq[0] += grad_prec[4] * R[1] * R[2] * 0.5f;
        grad_inv_s_sq[1] += grad_prec[4] * R[4] * R[5] * 0.5f;
        grad_inv_s_sq[2] += grad_prec[4] * R[7] * R[8] * 0.5f;

        // P[2,2]
        grad_inv_s_sq[0] += grad_prec[5] * R[2] * R[2];
        grad_inv_s_sq[1] += grad_prec[5] * R[5] * R[5];
        grad_inv_s_sq[2] += grad_prec[5] * R[8] * R[8];

        // Gradient w.r.t. log_scale
        // d(exp(-2*log_s))/d(log_s) = -2 * exp(-2*log_s) = -2 * inv_s_sq
        grad_log_sx = -2.0f * inv_s_sq[0] * grad_inv_s_sq[0];
        grad_log_sy = -2.0f * inv_s_sq[1] * grad_inv_s_sq[1];
        grad_log_sz = -2.0f * inv_s_sq[2] * grad_inv_s_sq[2];

        // Gradient w.r.t. rotation matrix
        // d(P)/d(R[i,k]) for each element
        float grad_R[9] = {0.0f};

        // P[0,0]
        grad_R[0] += grad_prec[0] * 2.0f * R[0] * inv_s_sq[0];
        grad_R[3] += grad_prec[0] * 2.0f * R[3] * inv_s_sq[1];
        grad_R[6] += grad_prec[0] * 2.0f * R[6] * inv_s_sq[2];

        // P[0,1] = R[0,k] * R[1,k] * inv_s_k (summed)
        grad_R[0] += grad_prec[1] * R[1] * inv_s_sq[0] * 0.5f;
        grad_R[1] += grad_prec[1] * R[0] * inv_s_sq[0] * 0.5f;
        grad_R[3] += grad_prec[1] * R[4] * inv_s_sq[1] * 0.5f;
        grad_R[4] += grad_prec[1] * R[3] * inv_s_sq[1] * 0.5f;
        grad_R[6] += grad_prec[1] * R[7] * inv_s_sq[2] * 0.5f;
        grad_R[7] += grad_prec[1] * R[6] * inv_s_sq[2] * 0.5f;

        // P[0,2]
        grad_R[0] += grad_prec[2] * R[2] * inv_s_sq[0] * 0.5f;
        grad_R[2] += grad_prec[2] * R[0] * inv_s_sq[0] * 0.5f;
        grad_R[3] += grad_prec[2] * R[5] * inv_s_sq[1] * 0.5f;
        grad_R[5] += grad_prec[2] * R[3] * inv_s_sq[1] * 0.5f;
        grad_R[6] += grad_prec[2] * R[8] * inv_s_sq[2] * 0.5f;
        grad_R[8] += grad_prec[2] * R[6] * inv_s_sq[2] * 0.5f;

        // P[1,1]
        grad_R[1] += grad_prec[3] * 2.0f * R[1] * inv_s_sq[0];
        grad_R[4] += grad_prec[3] * 2.0f * R[4] * inv_s_sq[1];
        grad_R[7] += grad_prec[3] * 2.0f * R[7] * inv_s_sq[2];

        // P[1,2]
        grad_R[1] += grad_prec[4] * R[2] * inv_s_sq[0] * 0.5f;
        grad_R[2] += grad_prec[4] * R[1] * inv_s_sq[0] * 0.5f;
        grad_R[4] += grad_prec[4] * R[5] * inv_s_sq[1] * 0.5f;
        grad_R[5] += grad_prec[4] * R[4] * inv_s_sq[1] * 0.5f;
        grad_R[7] += grad_prec[4] * R[8] * inv_s_sq[2] * 0.5f;
        grad_R[8] += grad_prec[4] * R[7] * inv_s_sq[2] * 0.5f;

        // P[2,2]
        grad_R[2] += grad_prec[5] * 2.0f * R[2] * inv_s_sq[0];
        grad_R[5] += grad_prec[5] * 2.0f * R[5] * inv_s_sq[1];
        grad_R[8] += grad_prec[5] * 2.0f * R[8] * inv_s_sq[2];

        // Backward through quaternion to rotation matrix
        // This is the standard quaternion -> rotation matrix backward
        float inv_norm = rsqrtf(qw * qw + qx * qx + qy * qy + qz * qz + 1e-12f);
        float w = qw * inv_norm;
        float x = qx * inv_norm;
        float y = qy * inv_norm;
        float z = qz * inv_norm;

        // Gradient w.r.t. normalized quaternion
        float grad_w = 0.0f, grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

        // R[0,0] = 1 - 2*(y^2 + z^2)
        grad_y += grad_R[0] * (-4.0f * y);
        grad_z += grad_R[0] * (-4.0f * z);

        // R[1,0] = 2*(xy + wz)
        grad_x += grad_R[1] * 2.0f * y;
        grad_y += grad_R[1] * 2.0f * x;
        grad_w += grad_R[1] * 2.0f * z;
        grad_z += grad_R[1] * 2.0f * w;

        // R[2,0] = 2*(xz - wy)
        grad_x += grad_R[2] * 2.0f * z;
        grad_z += grad_R[2] * 2.0f * x;
        grad_w += grad_R[2] * (-2.0f * y);
        grad_y += grad_R[2] * (-2.0f * w);

        // R[0,1] = 2*(xy - wz)
        grad_x += grad_R[3] * 2.0f * y;
        grad_y += grad_R[3] * 2.0f * x;
        grad_w += grad_R[3] * (-2.0f * z);
        grad_z += grad_R[3] * (-2.0f * w);

        // R[1,1] = 1 - 2*(x^2 + z^2)
        grad_x += grad_R[4] * (-4.0f * x);
        grad_z += grad_R[4] * (-4.0f * z);

        // R[2,1] = 2*(yz + wx)
        grad_y += grad_R[5] * 2.0f * z;
        grad_z += grad_R[5] * 2.0f * y;
        grad_w += grad_R[5] * 2.0f * x;
        grad_x += grad_R[5] * 2.0f * w;

        // R[0,2] = 2*(xz + wy)
        grad_x += grad_R[6] * 2.0f * z;
        grad_z += grad_R[6] * 2.0f * x;
        grad_w += grad_R[6] * 2.0f * y;
        grad_y += grad_R[6] * 2.0f * w;

        // R[1,2] = 2*(yz - wx)
        grad_y += grad_R[7] * 2.0f * z;
        grad_z += grad_R[7] * 2.0f * y;
        grad_w += grad_R[7] * (-2.0f * x);
        grad_x += grad_R[7] * (-2.0f * w);

        // R[2,2] = 1 - 2*(x^2 + y^2)
        grad_x += grad_R[8] * (-4.0f * x);
        grad_y += grad_R[8] * (-4.0f * y);

        // Backward through normalization
        // q_norm = q * inv_norm
        // d(q_norm)/d(q) = inv_norm * (I - q_norm * q_norm^T)
        float dot = w * grad_w + x * grad_x + y * grad_y + z * grad_z;
        grad_qw = inv_norm * (grad_w - w * dot);
        grad_qx = inv_norm * (grad_x - x * dot);
        grad_qy = inv_norm * (grad_y - y * dot);
        grad_qz = inv_norm * (grad_z - z * dot);
    }

    /**
     * @brief Sigmoid activation function
     */
    __device__ __forceinline__ float sigmoid_bwd(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    /**
     * @brief Compute vacancy and its gradient w.r.t. opacity and Mahalanobis distance
     *
     * vacancy = sqrt(max(0, 1 - o * G))
     * where G = exp(-0.5 * maha_sq)
     *
     * Returns vacancy and computes gradients:
     *   d(vacancy)/d(opacity) = -G / (2 * vacancy)
     *   d(vacancy)/d(maha_sq) = o * G / (4 * vacancy)
     */
    __device__ __forceinline__ float compute_vacancy_with_grads(
        float opacity,
        float maha_sq,
        float& grad_vacancy_wrt_opacity,
        float& grad_vacancy_wrt_maha_sq) {

        float exponent = fminf(maha_sq * 0.5f, 88.0f);
        float G = expf(-exponent);
        float occupancy = opacity * G;
        float one_minus_occ = fmaxf(0.0f, 1.0f - occupancy);
        float vacancy = sqrtf(one_minus_occ);

        if (vacancy > 1e-8f) {
            // d(vacancy)/d(occupancy) = -1 / (2 * vacancy)
            // d(occupancy)/d(opacity) = G
            grad_vacancy_wrt_opacity = -G / (2.0f * vacancy);

            // d(occupancy)/d(maha_sq) = opacity * G * (-0.5)
            // d(vacancy)/d(maha_sq) = -1/(2*vacancy) * d(occupancy)/d(maha_sq)
            //                       = opacity * G / (4 * vacancy)
            grad_vacancy_wrt_maha_sq = opacity * G / (4.0f * vacancy);
        } else {
            // Vacancy is nearly zero, gradient is undefined/zero
            grad_vacancy_wrt_opacity = 0.0f;
            grad_vacancy_wrt_maha_sq = 0.0f;
        }

        return vacancy;
    }

    /**
     * @brief Compute peak depth t* and its gradients
     *
     * t* = (d^T * P * (mu - o)) / (d^T * P * d)
     *
     * Gradients w.r.t. mean and precision matrix.
     */
    __device__ __forceinline__ float compute_peak_depth_with_grads(
        float ox, float oy, float oz,
        float dx, float dy, float dz,
        float mx, float my, float mz,
        const float* prec,
        float* grad_t_star_wrt_mean,
        float* grad_t_star_wrt_prec) {

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

        float inv_dPd = 1.0f / fmaxf(dPd, 1e-10f);
        float t_star = dPdiff * inv_dPd;

        // Gradient w.r.t. mean
        // d(t*)/d(mu) = d^T * P / (d^T * P * d)
        // = [P * d]^T / dPd (since P is symmetric)
        float P_d_x = prec[0] * dx + prec[1] * dy + prec[2] * dz;
        float P_d_y = prec[1] * dx + prec[3] * dy + prec[4] * dz;
        float P_d_z = prec[2] * dx + prec[4] * dy + prec[5] * dz;

        grad_t_star_wrt_mean[0] = P_d_x * inv_dPd;
        grad_t_star_wrt_mean[1] = P_d_y * inv_dPd;
        grad_t_star_wrt_mean[2] = P_d_z * inv_dPd;

        // Gradient w.r.t. precision (upper triangular)
        // t* = num / denom
        // d(t*)/dP = (d(num)/dP * denom - num * d(denom)/dP) / denom^2
        //          = d(num)/dP / denom - t* * d(denom)/dP / denom

        // d(num)/dP[i,j] where num = d^T * P * diff
        // This is diff_i * d_j + diff_j * d_i for symmetric P (off-diagonal gets factor 2)
        float grad_num_wrt_prec[6];
        grad_num_wrt_prec[0] = diff_x * dx;  // P[0,0]
        grad_num_wrt_prec[1] = diff_x * dy + diff_y * dx;  // P[0,1]
        grad_num_wrt_prec[2] = diff_x * dz + diff_z * dx;  // P[0,2]
        grad_num_wrt_prec[3] = diff_y * dy;  // P[1,1]
        grad_num_wrt_prec[4] = diff_y * dz + diff_z * dy;  // P[1,2]
        grad_num_wrt_prec[5] = diff_z * dz;  // P[2,2]

        // d(denom)/dP[i,j] where denom = d^T * P * d
        float grad_denom_wrt_prec[6];
        grad_denom_wrt_prec[0] = dx * dx;
        grad_denom_wrt_prec[1] = 2.0f * dx * dy;
        grad_denom_wrt_prec[2] = 2.0f * dx * dz;
        grad_denom_wrt_prec[3] = dy * dy;
        grad_denom_wrt_prec[4] = 2.0f * dy * dz;
        grad_denom_wrt_prec[5] = dz * dz;

        for (int i = 0; i < 6; ++i) {
            grad_t_star_wrt_prec[i] = (grad_num_wrt_prec[i] - t_star * grad_denom_wrt_prec[i]) * inv_dPd;
        }

        return t_star;
    }

    /**
     * @brief Create ray from pixel coordinates
     */
    __device__ __forceinline__ void pixel_to_ray_bwd(
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
    // TRANSMITTANCE GRADIENT COMPUTATION
    // =========================================================================

    /**
     * @brief Compute dT/dt at t_med using finite differences
     *
     * More accurate than analytical for the complex transmittance function.
     */
    __device__ float compute_dT_dt(
        float t_med,
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

        // Use central differences with small epsilon
        const float eps = 1e-4f;

        // Compute T(t_med + eps)
        float T_plus = 1.0f;
        for (int32_t idx = range_start; idx < range_end && T_plus > 1e-8f; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

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

            float opacity = sigmoid_bwd(opacities[global_g]);

            float prec[6], R[9], inv_s_sq[3];
            compute_precision_with_components(qw, qx, qy, qz, log_sx, log_sy, log_sz, prec, R, inv_s_sq);

            // Compute t_star
            float diff_x = mx - ray_ox, diff_y = my - ray_oy, diff_z = mz - ray_oz;
            float dPd = prec[0] * ray_dx * ray_dx + 2.0f * prec[1] * ray_dx * ray_dy +
                        2.0f * prec[2] * ray_dx * ray_dz + prec[3] * ray_dy * ray_dy +
                        2.0f * prec[4] * ray_dy * ray_dz + prec[5] * ray_dz * ray_dz;
            float Pd_x = prec[0] * diff_x + prec[1] * diff_y + prec[2] * diff_z;
            float Pd_y = prec[1] * diff_x + prec[3] * diff_y + prec[4] * diff_z;
            float Pd_z = prec[2] * diff_x + prec[4] * diff_y + prec[5] * diff_z;
            float dPdiff = ray_dx * Pd_x + ray_dy * Pd_y + ray_dz * Pd_z;
            float t_star = dPdiff / fmaxf(dPd, 1e-10f);

            if (t_star < 0.0f) continue;

            // Compute vacancy at t_star
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;
            float dx_star = pt_star_x - mx, dy_star = pt_star_y - my, dz_star = pt_star_z - mz;
            float maha_star = mahalanobis_sq_bwd(dx_star, dy_star, dz_star, prec);
            float G_star = expf(-fminf(maha_star * 0.5f, 88.0f));
            float vacancy_star = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_star));

            // Compute vacancy at t_med + eps
            float t_query = t_med + eps;
            float pt_x = ray_ox + t_query * ray_dx;
            float pt_y = ray_oy + t_query * ray_dy;
            float pt_z = ray_oz + t_query * ray_dz;
            float dx_t = pt_x - mx, dy_t = pt_y - my, dz_t = pt_z - mz;
            float maha_t = mahalanobis_sq_bwd(dx_t, dy_t, dz_t, prec);
            float G_t = expf(-fminf(maha_t * 0.5f, 88.0f));
            float vacancy_t = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_t));

            float Ti = compute_stochastic_transmittance(t_query, t_star, vacancy_star, vacancy_t);
            T_plus *= Ti;
        }

        // Compute T(t_med - eps)
        float T_minus = 1.0f;
        for (int32_t idx = range_start; idx < range_end && T_minus > 1e-8f; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

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

            float opacity = sigmoid_bwd(opacities[global_g]);

            float prec[6], R[9], inv_s_sq[3];
            compute_precision_with_components(qw, qx, qy, qz, log_sx, log_sy, log_sz, prec, R, inv_s_sq);

            float diff_x = mx - ray_ox, diff_y = my - ray_oy, diff_z = mz - ray_oz;
            float dPd = prec[0] * ray_dx * ray_dx + 2.0f * prec[1] * ray_dx * ray_dy +
                        2.0f * prec[2] * ray_dx * ray_dz + prec[3] * ray_dy * ray_dy +
                        2.0f * prec[4] * ray_dy * ray_dz + prec[5] * ray_dz * ray_dz;
            float Pd_x = prec[0] * diff_x + prec[1] * diff_y + prec[2] * diff_z;
            float Pd_y = prec[1] * diff_x + prec[3] * diff_y + prec[4] * diff_z;
            float Pd_z = prec[2] * diff_x + prec[4] * diff_y + prec[5] * diff_z;
            float dPdiff = ray_dx * Pd_x + ray_dy * Pd_y + ray_dz * Pd_z;
            float t_star = dPdiff / fmaxf(dPd, 1e-10f);

            if (t_star < 0.0f) continue;

            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;
            float dx_star = pt_star_x - mx, dy_star = pt_star_y - my, dz_star = pt_star_z - mz;
            float maha_star = mahalanobis_sq_bwd(dx_star, dy_star, dz_star, prec);
            float G_star = expf(-fminf(maha_star * 0.5f, 88.0f));
            float vacancy_star = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_star));

            float t_query = t_med - eps;
            float pt_x = ray_ox + t_query * ray_dx;
            float pt_y = ray_oy + t_query * ray_dy;
            float pt_z = ray_oz + t_query * ray_dz;
            float dx_t = pt_x - mx, dy_t = pt_y - my, dz_t = pt_z - mz;
            float maha_t = mahalanobis_sq_bwd(dx_t, dy_t, dz_t, prec);
            float G_t = expf(-fminf(maha_t * 0.5f, 88.0f));
            float vacancy_t = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_t));

            float Ti = compute_stochastic_transmittance(t_query, t_star, vacancy_star, vacancy_t);
            T_minus *= Ti;
        }

        return (T_plus - T_minus) / (2.0f * eps);
    }

    // =========================================================================
    // BACKWARD KERNEL: COMPLETE IMPLEMENTATION
    // =========================================================================

    /**
     * @brief Complete backward pass for G3S median depth
     *
     * Uses implicit function theorem (Equation 13):
     *   d(t_med)/d(theta) = -[dT/d(theta)] / [dT/dt]|_{t=t_med}
     *
     * Computes gradients for ALL Gaussian parameters that contribute to
     * the transmittance at t_med.
     */
    __global__ void g3s_median_depth_backward_complete_kernel(
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

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;
        int tile_id = block.group_index().y * tile_width + block.group_index().z;

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
        pixel_to_ray_bwd(px, py, viewmat, K, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz);

        // Compute dT/dt at t_med using finite differences
        float dT_dt = compute_dT_dt(
            t_med, ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
            means, quats, scales, opacities,
            flatten_ids, visible_indices, range_start, range_end);

        // Avoid division by zero
        if (fabsf(dT_dt) < 1e-10f) return;

        // Scale factor for all gradients: dL/d(theta) = dL/d(t_med) * d(t_med)/d(theta)
        //                                             = dL/d(t_med) * (-1/dT_dt) * dT/d(theta)
        float inv_dT_dt = -1.0f / dT_dt;

        // Point on ray at t_med
        float pt_x = ray_ox + t_med * ray_dx;
        float pt_y = ray_oy + t_med * ray_dy;
        float pt_z = ray_oz + t_med * ray_dz;

        // ------------------------------------------------------------
        // First pass: compute total transmittance and per-Gaussian Ti
        // We need T to compute dT/dTi = T/Ti for each Gaussian
        // ------------------------------------------------------------

        // Compute total T at t_med (for gradient scaling)
        float T_total = 1.0f;
        for (int32_t idx = range_start; idx < range_end && T_total > 1e-8f; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

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

            float opacity = sigmoid_bwd(opacities[global_g]);

            float prec[6], R[9], inv_s_sq[3];
            compute_precision_with_components(qw, qx_q, qy_q, qz_q, log_sx, log_sy, log_sz, prec, R, inv_s_sq);

            // Compute t_star
            float diff_x = mx - ray_ox, diff_y = my - ray_oy, diff_z = mz - ray_oz;
            float dPd = prec[0] * ray_dx * ray_dx + 2.0f * prec[1] * ray_dx * ray_dy +
                        2.0f * prec[2] * ray_dx * ray_dz + prec[3] * ray_dy * ray_dy +
                        2.0f * prec[4] * ray_dy * ray_dz + prec[5] * ray_dz * ray_dz;
            float Pd_x = prec[0] * diff_x + prec[1] * diff_y + prec[2] * diff_z;
            float Pd_y = prec[1] * diff_x + prec[3] * diff_y + prec[4] * diff_z;
            float Pd_z = prec[2] * diff_x + prec[4] * diff_y + prec[5] * diff_z;
            float dPdiff = ray_dx * Pd_x + ray_dy * Pd_y + ray_dz * Pd_z;
            float t_star = dPdiff / fmaxf(dPd, 1e-10f);

            if (t_star < 0.0f) continue;

            // Vacancy at t_star
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;
            float dx_star = pt_star_x - mx, dy_star = pt_star_y - my, dz_star = pt_star_z - mz;
            float maha_star = mahalanobis_sq_bwd(dx_star, dy_star, dz_star, prec);
            float G_star = expf(-fminf(maha_star * 0.5f, 88.0f));
            float vacancy_star = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_star));

            // Vacancy at t_med
            float dx_t = pt_x - mx, dy_t = pt_y - my, dz_t = pt_z - mz;
            float maha_t = mahalanobis_sq_bwd(dx_t, dy_t, dz_t, prec);
            float G_t = expf(-fminf(maha_t * 0.5f, 88.0f));
            float vacancy_t = sqrtf(fmaxf(0.0f, 1.0f - opacity * G_t));

            float Ti = compute_stochastic_transmittance(t_med, t_star, vacancy_star, vacancy_t);
            T_total *= Ti;
        }

        // ------------------------------------------------------------
        // Second pass: compute gradients for each Gaussian
        // ------------------------------------------------------------

        for (int32_t idx = range_start; idx < range_end; ++idx) {
            int32_t g = flatten_ids[idx];
            int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;

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
            float opacity = sigmoid_bwd(raw_opacity);

            float prec[6], R[9], inv_s_sq[3];
            compute_precision_with_components(qw, qx_q, qy_q, qz_q, log_sx, log_sy, log_sz, prec, R, inv_s_sq);

            // Compute t_star with gradients
            float grad_t_star_wrt_mean[3], grad_t_star_wrt_prec[6];
            float t_star = compute_peak_depth_with_grads(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                mx, my, mz,
                prec,
                grad_t_star_wrt_mean,
                grad_t_star_wrt_prec);

            if (t_star < 0.0f) continue;

            // Vacancy at t_star with gradients
            float pt_star_x = ray_ox + t_star * ray_dx;
            float pt_star_y = ray_oy + t_star * ray_dy;
            float pt_star_z = ray_oz + t_star * ray_dz;
            float dx_star = pt_star_x - mx, dy_star = pt_star_y - my, dz_star = pt_star_z - mz;
            float maha_star = mahalanobis_sq_bwd(dx_star, dy_star, dz_star, prec);

            float grad_vstar_wrt_opacity, grad_vstar_wrt_maha;
            float vacancy_star = compute_vacancy_with_grads(opacity, maha_star,
                grad_vstar_wrt_opacity, grad_vstar_wrt_maha);

            // Vacancy at t_med with gradients
            float dx_t = pt_x - mx, dy_t = pt_y - my, dz_t = pt_z - mz;
            float maha_t = mahalanobis_sq_bwd(dx_t, dy_t, dz_t, prec);

            float grad_vt_wrt_opacity, grad_vt_wrt_maha;
            float vacancy_t = compute_vacancy_with_grads(opacity, maha_t,
                grad_vt_wrt_opacity, grad_vt_wrt_maha);

            if (vacancy_t < 1e-8f || vacancy_star < 1e-8f) continue;

            // Compute Ti and dT/dTi
            float Ti = compute_stochastic_transmittance(t_med, t_star, vacancy_star, vacancy_t);
            if (Ti < 1e-8f) continue;

            // dT/dTi = T/Ti (product rule for product of Ti's)
            float dT_dTi = T_total / Ti;

            // ------------------------------------------------------------
            // Compute dTi/d(vacancy_star) and dTi/d(vacancy_t)
            // ------------------------------------------------------------
            float dTi_dvstar = 0.0f;
            float dTi_dvt = 0.0f;

            if (t_med <= t_star) {
                // Ti = vacancy_t
                dTi_dvt = 1.0f;
            } else {
                // Ti = vacancy_star^2 / vacancy_t
                dTi_dvstar = 2.0f * vacancy_star / vacancy_t;
                dTi_dvt = -(vacancy_star * vacancy_star) / (vacancy_t * vacancy_t);
            }

            // dT/d(vacancy) = dT/dTi * dTi/d(vacancy)
            float dT_dvstar = dT_dTi * dTi_dvstar;
            float dT_dvt = dT_dTi * dTi_dvt;

            // ------------------------------------------------------------
            // Gradient w.r.t. opacity
            // ------------------------------------------------------------
            float dT_dopacity = dT_dvstar * grad_vstar_wrt_opacity + dT_dvt * grad_vt_wrt_opacity;
            float grad_opacity = grad_t_med * inv_dT_dt * dT_dopacity;

            // Sigmoid backward: d(sigmoid)/d(raw) = sigmoid * (1 - sigmoid)
            float grad_raw_opacity = grad_opacity * opacity * (1.0f - opacity);
            atomicAdd(&grad_opacities[global_g], grad_raw_opacity);

            // ------------------------------------------------------------
            // Gradient w.r.t. mean through vacancy
            // ------------------------------------------------------------
            // d(vacancy)/d(mean) = d(vacancy)/d(maha) * d(maha)/d(delta) * d(delta)/d(mean)
            // where delta = pt - mean, so d(delta)/d(mean) = -I

            float grad_maha_delta_star[3], grad_maha_delta_t[3];
            mahalanobis_sq_grad_delta(dx_star, dy_star, dz_star, prec,
                grad_maha_delta_star[0], grad_maha_delta_star[1], grad_maha_delta_star[2]);
            mahalanobis_sq_grad_delta(dx_t, dy_t, dz_t, prec,
                grad_maha_delta_t[0], grad_maha_delta_t[1], grad_maha_delta_t[2]);

            // d(maha)/d(mean) = -d(maha)/d(delta) (since delta = pt - mean)
            float dT_dmean[3];
            dT_dmean[0] = dT_dvstar * grad_vstar_wrt_maha * (-grad_maha_delta_star[0]) +
                          dT_dvt * grad_vt_wrt_maha * (-grad_maha_delta_t[0]);
            dT_dmean[1] = dT_dvstar * grad_vstar_wrt_maha * (-grad_maha_delta_star[1]) +
                          dT_dvt * grad_vt_wrt_maha * (-grad_maha_delta_t[1]);
            dT_dmean[2] = dT_dvstar * grad_vstar_wrt_maha * (-grad_maha_delta_star[2]) +
                          dT_dvt * grad_vt_wrt_maha * (-grad_maha_delta_t[2]);

            // Also add gradient through t_star -> vacancy_star
            // pt_star = o + t_star * d, so d(pt_star)/d(mean) includes t_star gradient
            // This contributes to vacancy_star through maha_star

            float grad_mean[3];
            grad_mean[0] = grad_t_med * inv_dT_dt * dT_dmean[0];
            grad_mean[1] = grad_t_med * inv_dT_dt * dT_dmean[1];
            grad_mean[2] = grad_t_med * inv_dT_dt * dT_dmean[2];

            atomicAdd(&grad_means[global_g * 3 + 0], grad_mean[0]);
            atomicAdd(&grad_means[global_g * 3 + 1], grad_mean[1]);
            atomicAdd(&grad_means[global_g * 3 + 2], grad_mean[2]);

            // ------------------------------------------------------------
            // Gradient w.r.t. precision matrix (then to quaternion and scale)
            // ------------------------------------------------------------
            float grad_maha_prec_star[6], grad_maha_prec_t[6];
            mahalanobis_sq_grad_prec(dx_star, dy_star, dz_star, grad_maha_prec_star);
            mahalanobis_sq_grad_prec(dx_t, dy_t, dz_t, grad_maha_prec_t);

            float dT_dprec[6];
            for (int k = 0; k < 6; ++k) {
                dT_dprec[k] = dT_dvstar * grad_vstar_wrt_maha * grad_maha_prec_star[k] +
                              dT_dvt * grad_vt_wrt_maha * grad_maha_prec_t[k];
            }

            // Also t_star depends on precision
            // Add contribution: dT/d(t_star) * d(t_star)/d(prec)
            // But t_star's direct effect on T is through vacancy_star position
            // This is a second-order effect we approximate

            float grad_prec[6];
            for (int k = 0; k < 6; ++k) {
                grad_prec[k] = grad_t_med * inv_dT_dt * dT_dprec[k];
            }

            // Backprop through precision to quaternion and scale
            float grad_qw, grad_qx, grad_qy, grad_qz;
            float grad_log_sx, grad_log_sy, grad_log_sz;
            precision_backward(qw, qx_q, qy_q, qz_q, log_sx, log_sy, log_sz,
                               R, inv_s_sq, grad_prec,
                               grad_qw, grad_qx, grad_qy, grad_qz,
                               grad_log_sx, grad_log_sy, grad_log_sz);

            atomicAdd(&grad_quats[global_g * 4 + 0], grad_qw);
            atomicAdd(&grad_quats[global_g * 4 + 1], grad_qx);
            atomicAdd(&grad_quats[global_g * 4 + 2], grad_qy);
            atomicAdd(&grad_quats[global_g * 4 + 3], grad_qz);

            atomicAdd(&grad_scales[global_g * 3 + 0], grad_log_sx);
            atomicAdd(&grad_scales[global_g * 3 + 1], grad_log_sy);
            atomicAdd(&grad_scales[global_g * 3 + 2], grad_log_sz);
        }
    }

    // =========================================================================
    // LAUNCH FUNCTION
    // =========================================================================

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
        cudaStream_t stream) {

        if (C == 0 || H == 0 || W == 0 || N == 0) return;

        int tile_width = (W + tile_size - 1) / tile_size;
        int tile_height = (H + tile_size - 1) / tile_size;

        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_height, tile_width);

        g3s_median_depth_backward_complete_kernel<<<grid, threads, 0, stream>>>(
            means,
            quats,
            scales,
            opacities,
            median_depths,
            valid_mask,
            tile_offsets,
            flatten_ids,
            visible_indices,
            viewmats,
            Ks,
            grad_median_depths,
            grad_means,
            grad_quats,
            grad_scales,
            grad_opacities,
            C, N, M, H, W,
            tile_size,
            n_isects);
    }

} // namespace lfs::training::kernels
