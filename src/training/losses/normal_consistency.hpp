/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief G³S Normal Consistency Loss for Gaussian Splatting
     *
     * Implements the normal consistency loss from G³S (Geometry-Guided Gaussian
     * Splatting) that enforces consistency between Gaussian normals and depth-
     * derived surface normals.
     *
     * Loss formula:
     *   L_n = Σ ω_i · (1 - n_i · ñ)
     *
     * where:
     *   n_i = Gaussian normal (shortest axis of the ellipsoid)
     *   ñ   = normal from depth gradient (cross product of depth derivatives)
     *   ω_i = alpha blending weight for Gaussian i
     *
     * The loss is 0 when Gaussian normals align with depth-derived normals,
     * and 2 when they are opposite.
     *
     * NOTE: This loss writes gradients directly to input tensors in-place.
     */
    struct NormalConsistencyLoss {
        struct Params {
            float weight; ///< Loss weight multiplier
        };

        struct Context {
            lfs::core::Tensor loss_tensor;     ///< [1] scalar loss on GPU
            lfs::core::Tensor gaussian_normals; ///< [N, 3] computed Gaussian normals (for visualization)
            lfs::core::Tensor depth_normals;    ///< [H, W, 3] computed depth normals (for visualization)
        };

        /**
         * @brief Compute normal consistency loss and accumulate gradients
         *
         * @param quaternions [N, 4] unit quaternions (w, x, y, z) for each Gaussian
         * @param scales [N, 3] scale factors for each Gaussian (log space)
         * @param depth_map [H, W] rendered depth map
         * @param blend_weights [H, W, K] per-pixel Gaussian blend weights (K = max Gaussians per pixel)
         * @param gaussian_indices [H, W, K] indices of contributing Gaussians per pixel
         * @param camera_fx Camera focal length x
         * @param camera_fy Camera focal length y
         * @param params Loss parameters
         * @param quaternions_grad [N, 4] gradient tensor (accumulated in-place)
         * @param scales_grad [N, 3] gradient tensor (accumulated in-place)
         * @return (loss_tensor, context) or error - loss stays on GPU!
         */
        std::expected<std::pair<lfs::core::Tensor, Context>, std::string> forward(
            const lfs::core::Tensor& quaternions,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& depth_map,
            const lfs::core::Tensor& blend_weights,
            const lfs::core::Tensor& gaussian_indices,
            float camera_fx,
            float camera_fy,
            const Params& params,
            lfs::core::Tensor& quaternions_grad,
            lfs::core::Tensor& scales_grad);

    private:
        // Pre-allocated workspace tensors to eliminate allocation churn
        lfs::core::Tensor gaussian_normals_;   ///< [N, 3] Gaussian normals
        lfs::core::Tensor depth_normals_;      ///< [H, W, 3] Depth-derived normals
        lfs::core::Tensor loss_per_pixel_;     ///< [H, W] Per-pixel loss
        lfs::core::Tensor reduction_buffer_;   ///< Reduction workspace
        lfs::core::Tensor loss_scalar_;        ///< [1] Final scalar loss

        size_t allocated_n_ = 0;               ///< Track allocated Gaussian count
        size_t allocated_h_ = 0;               ///< Track allocated height
        size_t allocated_w_ = 0;               ///< Track allocated width
        size_t allocated_k_ = 0;               ///< Track allocated max Gaussians per pixel

        /**
         * @brief Ensure workspace buffers are allocated with correct sizes
         */
        void ensure_buffers(size_t n_gaussians, size_t height, size_t width, size_t k);
    };

    // ============================================================================
    // Helper functions for normal computation
    // ============================================================================

    /**
     * @brief Compute Gaussian normals from quaternions and scales
     *
     * The normal is the direction corresponding to the smallest scale (shortest axis).
     * For a Gaussian with quaternion q and scales s, the normal is the column of
     * the rotation matrix R = quat_to_rotmat(q) corresponding to argmin(s).
     *
     * @param quaternions [N, 4] unit quaternions (w, x, y, z)
     * @param scales [N, 3] scale factors (linear space, not log)
     * @return [N, 3] unit normal vectors
     */
    lfs::core::Tensor compute_gaussian_normals(
        const lfs::core::Tensor& quaternions,
        const lfs::core::Tensor& scales);

    /**
     * @brief Compute surface normals from depth map using finite differences
     *
     * Computes normals as n = normalize(∂z/∂x × ∂z/∂y) using Sobel-like
     * finite differences with camera intrinsics to account for perspective.
     *
     * @param depth_map [H, W] depth values
     * @param fx Camera focal length x
     * @param fy Camera focal length y
     * @return [H, W, 3] unit normal vectors (invalid regions set to [0, 0, 0])
     */
    lfs::core::Tensor compute_depth_normals(
        const lfs::core::Tensor& depth_map,
        float fx,
        float fy);

} // namespace lfs::training::losses
