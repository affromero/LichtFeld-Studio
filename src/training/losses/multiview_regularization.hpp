/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief Configuration for multi-view geometric regularization (G3S-style)
     *
     * Implements the geometric consistency loss from PGSR paper used in G3S:
     *   L_gc = sum_i w(u_i) * psi(u_i)
     * where:
     *   psi(u) = ||u - H_nr * H_rn * u||_2  (cycle reprojection error)
     *   w(u) = exp(-psi(u)) if psi < 1, else 0  (confidence weight)
     *   H_rn = homography from reference to neighbor using depth + normal
     *   H_nr = homography from neighbor back to reference
     */
    struct MultiViewRegConfig {
        float photometric_weight = 0.6f;  ///< Weight for NCC photometric consistency
        float geometric_weight = 0.02f;   ///< Weight for cycle reprojection error
        int patch_size = 11;              ///< NCC window size (must be odd)
        float depth_tolerance = 0.1f;     ///< Relative depth tolerance for valid correspondences
        float error_threshold = 1.0f;     ///< Max reprojection error for confidence weighting
        float min_depth = 0.01f;          ///< Minimum valid depth value
    };

    /**
     * @brief Multi-view geometric regularization loss for 3D Gaussian Splatting
     *
     * Enforces geometric consistency between rendered depth/normals and neighboring views.
     * Uses depth-induced homographies to warp reference pixels to neighbors and back,
     * penalizing cycle inconsistencies.
     *
     * Key operations:
     * 1. Compute depth-induced homography H_rn from reference to neighbor
     * 2. Warp reference pixels to neighbor view
     * 3. Compute inverse homography H_nr from neighbor back to reference
     * 4. Measure cycle reprojection error ||u - H_nr * H_rn * u||
     * 5. Weight by confidence (low error = high weight)
     * 6. Optional: NCC photometric consistency for patch matching
     */
    class MultiViewRegularization {
    public:
        /**
         * @brief Compute multi-view geometric consistency loss
         *
         * @param ref_cam Reference camera (provides intrinsics and extrinsics)
         * @param neighbor_cam Neighbor camera
         * @param ref_depth [H, W] rendered depth map from reference view
         * @param ref_normal [H, W, 3] rendered normal map from reference view (world space)
         * @param ref_image [H, W, 3] or [H, W, C] reference RGB image
         * @param neighbor_image [H, W, 3] or [H, W, C] neighbor RGB image
         * @param config Regularization parameters
         * @return Tensor with loss value [1] on GPU, or error string
         *
         * The loss combines:
         * - Geometric term: weighted cycle reprojection error
         * - Photometric term: NCC-based patch similarity
         */
        std::expected<lfs::core::Tensor, std::string> compute_loss(
            const lfs::core::Camera& ref_cam,
            const lfs::core::Camera& neighbor_cam,
            const lfs::core::Tensor& ref_depth,
            const lfs::core::Tensor& ref_normal,
            const lfs::core::Tensor& ref_image,
            const lfs::core::Tensor& neighbor_image,
            const MultiViewRegConfig& config = {});

        /**
         * @brief Compute loss with gradient accumulation
         *
         * @param ref_cam Reference camera
         * @param neighbor_cam Neighbor camera
         * @param ref_depth [H, W] rendered depth map
         * @param ref_normal [H, W, 3] rendered normal map
         * @param ref_image [H, W, 3] reference RGB image
         * @param neighbor_image [H, W, 3] neighbor RGB image
         * @param ref_depth_grad [H, W] gradient accumulator for depth
         * @param ref_normal_grad [H, W, 3] gradient accumulator for normals
         * @param config Regularization parameters
         * @return Loss tensor [1] on GPU, or error string
         */
        std::expected<lfs::core::Tensor, std::string> compute_loss_with_grad(
            const lfs::core::Camera& ref_cam,
            const lfs::core::Camera& neighbor_cam,
            const lfs::core::Tensor& ref_depth,
            const lfs::core::Tensor& ref_normal,
            const lfs::core::Tensor& ref_image,
            const lfs::core::Tensor& neighbor_image,
            lfs::core::Tensor& ref_depth_grad,
            lfs::core::Tensor& ref_normal_grad,
            const MultiViewRegConfig& config = {});

    private:
        // Pre-allocated workspace buffers to avoid repeated allocations
        lfs::core::Tensor warped_coords_;       // [H, W, 2] warped coordinates
        lfs::core::Tensor cycle_coords_;        // [H, W, 2] cycle-warped coordinates
        lfs::core::Tensor reprojection_error_;  // [H, W] per-pixel reprojection error
        lfs::core::Tensor confidence_weights_;  // [H, W] confidence weights
        lfs::core::Tensor ncc_scores_;          // [H, W] NCC patch matching scores
        lfs::core::Tensor loss_buffer_;         // [1] loss accumulator
        lfs::core::Tensor temp_reduction_;      // Temporary buffer for reductions

        // Track allocated dimensions for buffer reuse
        int allocated_height_ = 0;
        int allocated_width_ = 0;

        // Ensure workspace buffers are allocated for given dimensions
        void ensure_workspace(int height, int width);

        // Validate input tensors
        std::expected<void, std::string> validate_inputs(
            const lfs::core::Camera& ref_cam,
            const lfs::core::Camera& neighbor_cam,
            const lfs::core::Tensor& ref_depth,
            const lfs::core::Tensor& ref_normal,
            const lfs::core::Tensor& ref_image,
            const lfs::core::Tensor& neighbor_image);
    };

    // =============================================================================
    // Utility functions for homography computation
    // =============================================================================

    /**
     * @brief Compute depth-induced homography between two views
     *
     * For a planar patch at depth d with normal n, the homography from
     * reference to neighbor is:
     *   H = K_n * (R - t*n^T/d) * K_r^-1
     *
     * @param ref_cam Reference camera
     * @param neighbor_cam Neighbor camera
     * @param depth Depth value at the reference point
     * @param normal Surface normal at the reference point (world space)
     * @return 3x3 homography matrix
     */
    lfs::core::Tensor compute_depth_homography(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam,
        float depth,
        const lfs::core::Tensor& normal);

    /**
     * @brief Compute relative pose between two cameras
     *
     * @param ref_cam Reference camera
     * @param neighbor_cam Neighbor camera
     * @return Tuple of (R_rel, t_rel) where:
     *         R_rel = R_neighbor * R_reference^T
     *         t_rel = t_neighbor - R_rel * t_reference
     */
    std::pair<lfs::core::Tensor, lfs::core::Tensor> compute_relative_pose(
        const lfs::core::Camera& ref_cam,
        const lfs::core::Camera& neighbor_cam);

} // namespace lfs::training::losses
