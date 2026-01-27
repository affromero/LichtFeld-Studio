/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Compute vacancy field at arbitrary 3D query points
     *
     * Implements the G3S (Gaussian Geometry Guidance) vacancy field from:
     * "Gaussian Geometry Supervision for Enhanced Scene Reconstruction"
     *
     * Vacancy is defined as:
     *   v(x) = sqrt(1 - G(x))
     *
     * where G(x) is the Gaussian occupancy:
     *   G(x) = sum_i(o_i * exp(-(x - x_c_i)^T * Sigma_i^-1 * (x - x_c_i)))
     *
     * The covariance matrix Sigma is computed from quaternion and scales:
     *   Sigma = R * S * S^T * R^T
     *
     * where R is the rotation matrix from quaternion and S is the diagonal
     * scale matrix (with exp(log_scales) on diagonal).
     *
     * @param means Gaussian centers [N, 3] in world coordinates
     * @param quats Quaternions [N, 4] in (w, x, y, z) format
     * @param scales Log-scales [N, 3] (will be exponentiated internally)
     * @param opacities Log-opacities [N] (will be sigmoidified internally)
     * @param query_pts Query points [M, 3] in world coordinates
     * @param vacancy_out Output vacancy values [M] in range [0, 1]
     * @param N Number of Gaussians
     * @param M Number of query points
     * @param stream CUDA stream for async execution
     */
    void launch_compute_vacancy(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* query_pts,
        float* vacancy_out,
        int N,
        int M,
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute vacancy field with culling radius optimization
     *
     * Same as launch_compute_vacancy but skips Gaussians that are
     * further than cull_radius from the query point. This is a significant
     * optimization for large scenes.
     *
     * @param means Gaussian centers [N, 3] in world coordinates
     * @param quats Quaternions [N, 4] in (w, x, y, z) format
     * @param scales Log-scales [N, 3] (will be exponentiated internally)
     * @param opacities Log-opacities [N] (will be sigmoidified internally)
     * @param query_pts Query points [M, 3] in world coordinates
     * @param vacancy_out Output vacancy values [M] in range [0, 1]
     * @param N Number of Gaussians
     * @param M Number of query points
     * @param cull_radius Skip Gaussians further than this distance (in world units)
     * @param stream CUDA stream for async execution
     */
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
        cudaStream_t stream = nullptr);

    /**
     * @brief Compute vacancy gradient w.r.t. query points
     *
     * Backward pass for vacancy computation. Computes dL/d(query_pts)
     * given dL/d(vacancy_out).
     *
     * @param means Gaussian centers [N, 3]
     * @param quats Quaternions [N, 4]
     * @param scales Log-scales [N, 3]
     * @param opacities Log-opacities [N]
     * @param query_pts Query points [M, 3]
     * @param grad_vacancy Gradient w.r.t vacancy output [M]
     * @param grad_query_pts Output gradient w.r.t query points [M, 3]
     * @param N Number of Gaussians
     * @param M Number of query points
     * @param stream CUDA stream
     */
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
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
