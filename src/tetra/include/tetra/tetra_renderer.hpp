/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "tetra/tetra_features.hpp"
#include "tetra/tetra_mesh.hpp"

#include <expected>
#include <memory>
#include <string>

namespace lfs::tetra {

/**
 * @brief Configuration for tetrahedral renderer
 */
struct TetraRenderConfig {
    int tile_size = 16;              // Tile size for tiled rasterization
    int max_tets_per_tile = 256;     // Maximum tetrahedra per tile
    int max_intersections_per_pixel = 4;   // Maximum intersections per pixel (max_K) - reduced for memory
    float near_clip = 0.01f;         // Near clipping plane
    float far_clip = 1000.0f;        // Far clipping plane
    bool use_alpha_blending = true;  // Front-to-back alpha blending
    bool compute_depth = true;       // Compute depth buffer
    float alpha_threshold = 0.999f;  // Alpha threshold for early termination
};

/**
 * @brief Result of forward rendering pass
 */
struct RenderForwardResult {
    core::Tensor rgb;              // [H, W, 3] Rendered RGB image
    core::Tensor alpha;            // [H, W, 1] Alpha channel
    core::Tensor depth;            // [H, W, 1] Depth buffer (optional)
    core::Tensor num_intersects;   // [H, W] Number of tet intersections per pixel

    // For backward pass
    core::Tensor intersection_ids;     // [H, W, max_K] Tet indices per pixel
    core::Tensor barycentric_coords;   // [H, W, max_K, 4] Barycentric coordinates
    core::Tensor ray_depths;           // [H, W, max_K] Intersection depths

    // Per-sample intermediates for alpha_blend_backward
    core::Tensor sample_rgb;           // [H, W, max_K, 3] RGB per intersection
    core::Tensor sample_alpha;         // [H, W, max_K] Alpha per intersection

    // For neural network backward (hash grid + MLP training)
    core::Tensor tet_centroids;        // [T, 3] Tetrahedron centroids used for network query
    core::Tensor view_directions;      // [T, 3] View directions used for network query
    bool used_network = false;         // Whether network was used (vs per-tet fallback)
};

/**
 * @brief Gradients from backward rendering pass
 */
struct RenderBackwardResult {
    core::Tensor grad_vertices;    // [V, 3] Gradient w.r.t vertex positions
    core::Tensor grad_features;    // [N_query, F] Gradient w.r.t feature outputs

    // For feature network backward
    core::Tensor query_positions;  // [N_query, 3] Query positions used
    core::Tensor query_directions; // [N_query, 3] View directions used

    // Gradients for per-tetrahedron color parameters
    core::Tensor grad_density;     // [T] Gradient w.r.t density
    core::Tensor grad_base_color;  // [T, 3] Gradient w.r.t base color
    core::Tensor grad_gradient;    // [T, 3] Gradient w.r.t color gradient
};

/**
 * @brief Tile-based rasterizer for tetrahedral meshes
 *
 * Implements the rendering pipeline from Radiance Meshes:
 * 1. Project tetrahedra to screen space
 * 2. Tile-based binning for efficient rasterization
 * 3. Ray-tetrahedron intersection using Plucker coordinates
 * 4. Front-to-back alpha blending with feature evaluation
 *
 * The renderer supports both training (with gradients) and inference modes.
 */
class TetraRenderer {
public:
    TetraRenderer() = default;
    ~TetraRenderer();

    // Delete copy operations
    TetraRenderer(const TetraRenderer&) = delete;
    TetraRenderer& operator=(const TetraRenderer&) = delete;

    // Allow move operations
    TetraRenderer(TetraRenderer&&) noexcept;
    TetraRenderer& operator=(TetraRenderer&&) noexcept;

    // ------------------------------
    // INITIALIZATION
    // ------------------------------

    /**
     * @brief Initialize renderer with configuration
     * @param config Render configuration
     * @return Error on failure
     */
    std::expected<void, std::string> initialize(const TetraRenderConfig& config);

    // ------------------------------
    // FORWARD PASS
    // ------------------------------

    /**
     * @brief Render tetrahedral mesh
     *
     * @param mesh Tetrahedral mesh data
     * @param features Feature network for color computation (non-const for caching)
     * @param camera Camera parameters
     * @param background Background color [3] or image [H, W, 3]
     * @return RenderForwardResult with rendered image and auxiliary data
     */
    std::expected<RenderForwardResult, std::string> forward(
        const TetraMesh& mesh,
        TetraFeatures& features,
        const core::Camera& camera,
        const core::Tensor& background);

    /**
     * @brief Render with precomputed intersections (for training)
     *
     * Skips intersection computation, useful when mesh hasn't changed.
     *
     * @param mesh Tetrahedral mesh data
     * @param features Feature network (non-const for caching)
     * @param camera Camera parameters
     * @param background Background color/image
     * @param cached_intersections Previous forward result
     * @return RenderForwardResult with rendered image
     */
    std::expected<RenderForwardResult, std::string> forward_with_cache(
        const TetraMesh& mesh,
        TetraFeatures& features,
        const core::Camera& camera,
        const core::Tensor& background,
        const RenderForwardResult& cached_intersections);

    // ------------------------------
    // BACKWARD PASS
    // ------------------------------

    /**
     * @brief Compute gradients through rendering
     *
     * @param mesh Tetrahedral mesh data
     * @param features Feature network (non-const for backward pass)
     * @param camera Camera parameters
     * @param forward_result Result from forward pass
     * @param grad_rgb [H, W, 3] Gradient w.r.t output RGB
     * @param grad_alpha [H, W, 1] Gradient w.r.t output alpha (optional)
     * @return RenderBackwardResult with gradients
     */
    std::expected<RenderBackwardResult, std::string> backward(
        const TetraMesh& mesh,
        TetraFeatures& features,
        const core::Camera& camera,
        const RenderForwardResult& forward_result,
        const core::Tensor& grad_rgb,
        const core::Tensor& grad_alpha);

    // ------------------------------
    // CONFIGURATION
    // ------------------------------

    [[nodiscard]] const TetraRenderConfig& config() const { return config_; }

    void set_config(const TetraRenderConfig& config) { config_ = config; }

    // ------------------------------
    // UTILITIES
    // ------------------------------

    /**
     * @brief Compute per-tetrahedron visibility scores
     *
     * Used for mesh extraction - tetrahedra visible from training views
     * are more likely to be valid geometry.
     *
     * @param mesh Tetrahedral mesh
     * @param cameras Training cameras
     * @return Visibility scores [T] per tetrahedron
     */
    static core::Tensor compute_visibility_scores(
        const TetraMesh& mesh,
        const std::vector<core::Camera>& cameras);

private:
    TetraRenderConfig config_;

    // Workspace buffers (reused across frames)
    core::Tensor projected_verts_;     // [V, 4] Homogeneous screen coords
    core::Tensor tet_bounds_;          // [T, 4] Screen-space bounding boxes
    core::Tensor tile_ranges_;         // [num_tiles_x, num_tiles_y, 2] Tet ranges per tile
    core::Tensor sorted_tet_ids_;      // [T] Depth-sorted tetrahedra per tile

    // For intersection computation
    core::Tensor plucker_coords_;      // [T, 4, 6] Plucker edge coordinates

    bool initialized_ = false;

    // Internal methods
    std::expected<void, std::string> project_vertices(
        const TetraMesh& mesh,
        const core::Camera& camera);

    std::expected<void, std::string> compute_tile_assignments(
        const TetraMesh& mesh,
        int width, int height);
};

// ------------------------------
// CUDA KERNEL DECLARATIONS
// ------------------------------

namespace cuda {

    /**
     * @brief Ray-tetrahedron intersection using Plucker coordinates
     *
     * @param ray_origins [N, 3] Ray origins
     * @param ray_dirs [N, 3] Ray directions
     * @param vertices [V, 3] Vertex positions
     * @param tetrahedra [T, 4] Tetrahedron indices
     * @param hit_mask [N, T] Output: hit mask (bool)
     * @param barycentrics [N, T, 4] Output: barycentric coordinates
     * @param depths [N, T] Output: intersection depths
     * @param N Number of rays
     * @param T Number of tetrahedra
     * @param stream CUDA stream
     */
    void launch_ray_tet_intersection(
        const float* ray_origins,
        const float* ray_dirs,
        const float* vertices,
        const int64_t* tetrahedra,
        uint8_t* hit_mask,
        float* barycentrics,
        float* depths,
        size_t N,
        size_t T,
        void* stream = nullptr);

    /**
     * @brief Tile-based alpha blending for tetrahedra
     *
     * @param rgb [H, W, max_K, 3] Per-intersection RGB values
     * @param alpha [H, W, max_K] Per-intersection alpha values
     * @param depths [H, W, max_K] Per-intersection depths
     * @param num_intersects [H, W] Number of intersections per pixel
     * @param background [3] or [H, W, 3] Background color
     * @param output_rgb [H, W, 3] Output: blended RGB
     * @param output_alpha [H, W, 1] Output: accumulated alpha
     * @param width Image width
     * @param height Image height
     * @param max_K Maximum intersections per pixel
     * @param alpha_threshold Early termination threshold
     * @param stream CUDA stream
     */
    void launch_alpha_blend(
        const float* rgb,
        const float* alpha,
        const float* depths,
        const int32_t* num_intersects,
        const float* background,
        float* output_rgb,
        float* output_alpha,
        int width,
        int height,
        int max_K,
        float alpha_threshold,
        bool background_is_image,
        void* stream = nullptr);

    /**
     * @brief Backward pass for alpha blending
     */
    void launch_alpha_blend_backward(
        const float* rgb,
        const float* alpha,
        const float* depths,
        const int32_t* num_intersects,
        const float* grad_output_rgb,
        const float* grad_output_alpha,
        float* grad_rgb,
        float* grad_alpha,
        int width,
        int height,
        int max_K,
        float alpha_threshold,
        void* stream = nullptr);

    /**
     * @brief Generate rays from camera intrinsics and extrinsics
     *
     * For each pixel (px, py), computes:
     * - Ray direction in camera space: dir_cam = [(px - cx)/fx, (py - cy)/fy, 1.0]
     * - Transform to world space: dir_world = R^T @ dir_cam (normalized)
     * - Origin in world space: R^T @ (-T)
     *
     * @param ray_origins [H*W, 3] Output: camera position in world space
     * @param ray_directions [H*W, 3] Output: normalized ray directions in world space
     * @param R [3, 3] Rotation matrix (world-to-camera, row-major)
     * @param T [3] Translation vector (world-to-camera)
     * @param fx Focal length x
     * @param fy Focal length y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param width Image width
     * @param height Image height
     * @param stream CUDA stream
     */
    void launch_generate_rays(
        float* ray_origins,
        float* ray_directions,
        const float* R,
        const float* T,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        void* stream = nullptr);

    /**
     * @brief Generate rays with precomputed camera origin (optimized)
     *
     * When the camera origin is already computed, this version avoids
     * redundant computation per pixel. Useful for multi-pass rendering.
     *
     * @param ray_origins [H*W, 3] Output: camera position (repeated)
     * @param ray_directions [H*W, 3] Output: normalized ray directions
     * @param R [3, 3] Rotation matrix (world-to-camera, row-major)
     * @param origin_x Precomputed camera origin X
     * @param origin_y Precomputed camera origin Y
     * @param origin_z Precomputed camera origin Z
     * @param fx Focal length x
     * @param fy Focal length y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param width Image width
     * @param height Image height
     * @param stream CUDA stream
     */
    void launch_generate_rays_shared_origin(
        float* ray_origins,
        float* ray_directions,
        const float* R,
        float origin_x,
        float origin_y,
        float origin_z,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        void* stream = nullptr);

    /**
     * @brief Generate rays for a batch of pixel coordinates
     *
     * Generates rays only for specified pixels, useful for:
     * - Sparse sampling during training
     * - Adaptive ray marching
     * - Progressive rendering
     *
     * @param ray_origins [N, 3] Output: camera positions
     * @param ray_directions [N, 3] Output: normalized ray directions
     * @param pixel_coords [N, 2] Input: pixel coordinates (px, py)
     * @param R [3, 3] Rotation matrix (world-to-camera, row-major)
     * @param T [3] Translation vector
     * @param fx Focal length x
     * @param fy Focal length y
     * @param cx Principal point x
     * @param cy Principal point y
     * @param N Number of rays to generate
     * @param stream CUDA stream
     */
    void launch_generate_rays_batch(
        float* ray_origins,
        float* ray_directions,
        const int* pixel_coords,
        const float* R,
        const float* T,
        float fx,
        float fy,
        float cx,
        float cy,
        int N,
        void* stream = nullptr);

    /**
     * @brief Collect intersections from full hit_mask into compact per-pixel arrays
     *
     * Scans through the [N, T] hit_mask and collects up to max_K intersections
     * per ray, applying near/far clipping.
     *
     * @param hit_mask [N, T] Input: hit mask from ray_tet_intersection
     * @param all_depths [N, T] Input: depths from ray_tet_intersection
     * @param all_barycentrics [N, T, 4] Input: barycentrics from ray_tet_intersection
     * @param out_tet_indices [N, max_K] Output: collected tet indices (-1 for invalid)
     * @param out_depths [N, max_K] Output: collected depths
     * @param out_barycentrics [N, max_K, 4] Output: collected barycentrics
     * @param out_num_intersects [N] Output: count of valid intersections per ray
     * @param N Number of rays
     * @param T Number of tetrahedra
     * @param max_K Maximum intersections to collect per ray
     * @param near_clip Near clipping distance
     * @param far_clip Far clipping distance
     * @param stream CUDA stream
     */
    void launch_collect_intersections(
        const uint8_t* hit_mask,
        const float* all_depths,
        const float* all_barycentrics,
        int64_t* out_tet_indices,
        float* out_depths,
        float* out_barycentrics,
        int32_t* out_num_intersects,
        size_t N,
        size_t T,
        int max_K,
        float near_clip,
        float far_clip,
        void* stream = nullptr);

    /**
     * @brief Tile-based ray-tet intersection with direct collection
     *
     * Optimized version that uses screen-space bounding boxes to cull
     * tetrahedra and directly outputs compacted intersection arrays.
     * Much more memory-efficient than full NxT intersection for large meshes.
     *
     * @param ray_origins [H*W, 3] Ray origins
     * @param ray_dirs [H*W, 3] Ray directions
     * @param vertices [V, 3] Vertex positions
     * @param tetrahedra [T, 4] Tetrahedron indices
     * @param tet_bounds [T, 4] Screen-space bounding boxes (min_x, min_y, max_x, max_y)
     * @param out_tet_indices [H*W, max_K] Output: tet indices per pixel
     * @param out_depths [H*W, max_K] Output: intersection depths (sorted)
     * @param out_barycentrics [H*W, max_K, 4] Output: barycentric coordinates
     * @param out_num_intersects [H*W] Output: count per pixel
     * @param width Image width
     * @param height Image height
     * @param T Number of tetrahedra
     * @param max_K Maximum intersections per pixel
     * @param near_clip Near clipping distance
     * @param far_clip Far clipping distance
     * @param stream CUDA stream
     */
    void launch_ray_tet_intersection_tiled(
        const float* ray_origins,
        const float* ray_dirs,
        const float* vertices,
        const int64_t* tetrahedra,
        const float* tet_bounds,
        int64_t* out_tet_indices,
        float* out_depths,
        float* out_barycentrics,
        int32_t* out_num_intersects,
        int width,
        int height,
        size_t T,
        int max_K,
        float near_clip,
        float far_clip,
        void* stream = nullptr);

    /**
     * @brief Compute RGB and alpha for intersections using per-tet linear color model
     *
     * Evaluates the linear color model at each intersection point:
     *   color = max(base_color + dot(gradient, position), 0)
     *   alpha = 1 - exp(-exp(density) * segment_length)
     *
     * This matches the radiance_meshes Python implementation.
     *
     * @param vertices [V, 3] Vertex positions
     * @param tetrahedra [T, 4] Tetrahedron indices
     * @param tet_indices [H*W, max_K] Intersection tet indices
     * @param barycentrics [H*W, max_K, 4] Barycentric coordinates
     * @param depths [H*W, max_K] Intersection depths
     * @param num_intersects [H*W] Count per pixel
     * @param tet_density [T] Log-scale density per tetrahedron
     * @param tet_base_color [T, 3] Base color per tetrahedron
     * @param tet_gradient [T, 3] Color gradient per tetrahedron
     * @param out_rgb [H*W, max_K, 3] Output: RGB per intersection
     * @param out_alpha [H*W, max_K] Output: alpha per intersection
     * @param width Image width
     * @param height Image height
     * @param max_K Maximum intersections per pixel
     * @param use_network_colors If true, use base_color directly as RGB (no softplus)
     * @param stream CUDA stream
     */
    void launch_compute_intersection_colors(
        const float* vertices,
        const int64_t* tetrahedra,
        const int64_t* tet_indices,
        const float* barycentrics,
        const float* depths,
        const int32_t* num_intersects,
        const float* tet_density,
        const float* tet_base_color,
        const float* tet_gradient,
        float* out_rgb,
        float* out_alpha,
        int width,
        int height,
        int max_K,
        bool use_network_colors = false,
        void* stream = nullptr);

    /**
     * @brief Compute depth buffer from first intersection
     *
     * @param depths [H*W, max_K] Intersection depths (sorted)
     * @param num_intersects [H*W] Count per pixel
     * @param depth_buffer [H*W] Output: depth buffer (first hit)
     * @param width Image width
     * @param height Image height
     * @param max_K Maximum intersections per pixel
     * @param stream CUDA stream
     */
    void launch_compute_depth_buffer(
        const float* depths,
        const int32_t* num_intersects,
        float* depth_buffer,
        int width,
        int height,
        int max_K,
        void* stream = nullptr);

    /**
     * @brief Backward pass for per-tet color computation
     *
     * Computes gradients for density, base_color, and gradient parameters
     * based on the per-sample RGB and alpha gradients from alpha_blend_backward.
     *
     * @param vertices [V, 3] Vertex positions
     * @param tetrahedra [T, 4] Tetrahedron indices
     * @param tet_indices [H*W, max_K] Intersection tet indices
     * @param barycentrics [H*W, max_K, 4] Barycentric coordinates
     * @param depths [H*W, max_K] Intersection depths
     * @param num_intersects [H*W] Count per pixel
     * @param tet_density [T] Log-scale density per tetrahedron
     * @param tet_base_color [T, 3] Base color per tetrahedron
     * @param tet_gradient [T, 3] Color gradient per tetrahedron
     * @param grad_rgb [H*W, max_K, 3] Input: gradient w.r.t sample RGB
     * @param grad_alpha [H*W, max_K] Input: gradient w.r.t sample alpha
     * @param grad_density [T] Output: gradient w.r.t density
     * @param grad_base_color [T, 3] Output: gradient w.r.t base color
     * @param grad_gradient [T, 3] Output: gradient w.r.t color gradient
     * @param width Image width
     * @param height Image height
     * @param max_K Maximum intersections per pixel
     * @param stream CUDA stream
     */
    void launch_compute_intersection_colors_backward(
        const float* vertices,
        const int64_t* tetrahedra,
        const int64_t* tet_indices,
        const float* barycentrics,
        const float* depths,
        const int32_t* num_intersects,
        const float* tet_density,
        const float* tet_base_color,
        const float* tet_gradient,
        const float* grad_rgb,
        const float* grad_alpha,
        float* grad_density,
        float* grad_base_color,
        float* grad_gradient,
        int width,
        int height,
        int max_K,
        int64_t num_tets,
        int64_t num_vertices,
        void* stream = nullptr);

    /**
     * @brief Sort per-pixel intersections by depth
     *
     * @param depths [H*W, max_K] In/out: intersection depths
     * @param tet_indices [H*W, max_K] In/out: tet indices
     * @param barycentrics [H*W, max_K, 4] In/out: barycentric coordinates
     * @param num_intersects [H*W] Count per pixel
     * @param width Image width
     * @param height Image height
     * @param max_K Maximum intersections per pixel
     * @param stream CUDA stream
     */
    void launch_sort_intersections(
        float* depths,
        int64_t* tet_indices,
        float* barycentrics,
        const int32_t* num_intersects,
        int width,
        int height,
        int max_K,
        void* stream = nullptr);

} // namespace cuda

} // namespace lfs::tetra
