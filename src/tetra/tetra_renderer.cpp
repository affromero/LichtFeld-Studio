/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_renderer.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace lfs::tetra {

// CUDA kernel declarations
namespace cuda {
    // Tile-based rasterization functions
    void launch_count_tiles_per_tet(
        const float* tet_bounds,
        int32_t* tiles_per_tet,
        size_t num_tets,
        int tile_size,
        int tile_width,
        int tile_height,
        int image_width,
        int image_height,
        void* stream);

    void launch_generate_tet_tile_pairs(
        const float* tet_bounds,
        const float* tet_depths,
        const int64_t* cum_tiles,
        int64_t* isect_keys,
        int32_t* tet_indices,
        size_t num_tets,
        int tile_size,
        int tile_width,
        int tile_height,
        void* stream);

    void launch_compute_tile_ranges(
        const int64_t* sorted_keys,
        int32_t* tile_ranges,
        int64_t total_pairs,
        int num_tiles,
        void* stream);

    void launch_ray_tet_intersection_with_tiles(
        const float* ray_origins,
        const float* ray_dirs,
        const float* vertices,
        const int64_t* tetrahedra,
        const int32_t* sorted_tet_ids,
        const int32_t* tile_ranges,
        int64_t* out_tet_indices,
        float* out_depths,
        float* out_barycentrics,
        int32_t* out_num_intersects,
        int width,
        int height,
        int tile_size,
        int tile_width,
        int max_K,
        float near_clip,
        float far_clip,
        void* stream);

    void launch_compute_tet_depths(
        const float* vertices,
        const int64_t* tetrahedra,
        const float* R,
        const float* T_vec,
        float* depths,
        size_t num_tets,
        void* stream);

    void radix_sort_tet_tiles(
        int64_t n_pairs,
        int64_t* keys_in,
        int32_t* values_in,
        int64_t* keys_out,
        int32_t* values_out,
        int num_key_bits,
        void* stream);

    void compute_prefix_sum_int32_to_int64(
        const int32_t* input,
        int64_t* output,
        size_t n_elements,
        void* stream);

    int64_t get_last_prefix_sum_value(
        const int64_t* prefix_sum,
        size_t n_elements,
        void* stream);

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
        void* stream);

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
        void* stream);

    void launch_sort_intersections(
        float* depths,
        int64_t* tet_indices,
        float* barycentrics,
        const int32_t* num_intersects,
        int width,
        int height,
        int max_K,
        void* stream);

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
        void* stream);

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
        void* stream);

    void launch_compute_intersection_colors(
        const float* vertices,
        const int64_t* tetrahedra,
        const int64_t* tet_indices,
        const float* barycentrics,
        const float* depths,
        const int32_t* num_intersects,
        const float* tet_density,      // NEW
        const float* tet_base_color,   // NEW
        const float* tet_gradient,     // NEW
        float* out_rgb,
        float* out_alpha,
        int width,
        int height,
        int max_K,
        void* stream);

    void launch_compute_depth_buffer(
        const float* depths,
        const int32_t* num_intersects,
        float* depth_buffer,
        int width,
        int height,
        int max_K,
        void* stream);

    void launch_compute_tet_centroids(
        const float* vertices,
        const int64_t* tetrahedra,
        float* centroids,
        size_t num_tets,
        void* stream);

    void launch_project_vertices(
        const float* vertices,
        const float* R,
        const float* T,
        const float* intrinsics,
        float* projected,
        size_t num_vertices,
        float near_clip,
        void* stream);

    void launch_compute_tet_bounds(
        const int64_t* tetrahedra,
        const float* projected,
        float* bounds,
        size_t num_tets,
        void* stream);

    void launch_reconstruct_intersections(
        const float* vertices,
        const int64_t* tetrahedra,
        const float* barycentrics,
        const int64_t* tet_indices,
        const int32_t* num_intersects,
        const float* cam_pos,
        float* positions,
        float* directions,
        const int32_t* prefix_sum,
        int width,
        int height,
        int max_K,
        void* stream);

    void launch_prefix_sum(
        const int32_t* counts,
        int32_t* prefix_sum,
        size_t n,
        void* stream);

    void launch_propagate_gradients(
        const float* grad_rgb_hwc,
        const int32_t* num_intersects,
        const int32_t* prefix_sum,
        float* grad_features,
        int width,
        int height,
        int max_K,
        void* stream);

    void launch_vertex_gradients(
        const float* grad_features,
        const float* barycentrics,
        const int64_t* tet_indices,
        const int64_t* tetrahedra,
        const int32_t* num_intersects,
        const int32_t* prefix_sum,
        float* grad_vertices,
        int width,
        int height,
        int max_K,
        size_t num_vertices,
        void* stream);

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
        void* stream);
}

// ------------------------------
// LIFECYCLE
// ------------------------------

TetraRenderer::~TetraRenderer() = default;

TetraRenderer::TetraRenderer(TetraRenderer&& other) noexcept
    : config_(other.config_)
    , projected_verts_(std::move(other.projected_verts_))
    , tet_bounds_(std::move(other.tet_bounds_))
    , tile_ranges_(std::move(other.tile_ranges_))
    , sorted_tet_ids_(std::move(other.sorted_tet_ids_))
    , plucker_coords_(std::move(other.plucker_coords_))
    , initialized_(other.initialized_) {
    other.initialized_ = false;
}

TetraRenderer& TetraRenderer::operator=(TetraRenderer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        projected_verts_ = std::move(other.projected_verts_);
        tet_bounds_ = std::move(other.tet_bounds_);
        tile_ranges_ = std::move(other.tile_ranges_);
        sorted_tet_ids_ = std::move(other.sorted_tet_ids_);
        plucker_coords_ = std::move(other.plucker_coords_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

// ------------------------------
// INITIALIZATION
// ------------------------------

std::expected<void, std::string> TetraRenderer::initialize(const TetraRenderConfig& config) {
    config_ = config;
    initialized_ = true;
    return {};
}

// ------------------------------
// FORWARD PASS
// ------------------------------

std::expected<RenderForwardResult, std::string> TetraRenderer::forward(
    const TetraMesh& mesh,
    TetraFeatures& features,
    const core::Camera& camera,
    const core::Tensor& background) {

    if (!initialized_) {
        return std::unexpected("TetraRenderer not initialized");
    }

    if (mesh.num_tetrahedra() == 0) {
        return std::unexpected("Mesh has no tetrahedra");
    }

    const int width = camera.image_width() > 0 ? camera.image_width() : camera.camera_width();
    const int height = camera.image_height() > 0 ? camera.image_height() : camera.camera_height();
    const size_t num_pixels = static_cast<size_t>(width * height);
    const size_t num_tets = mesh.num_tetrahedra();
    const int max_K = config_.max_intersections_per_pixel;

    // ------------------------------------------------------------
    // Step 1: Project vertices to screen space (for bounding box culling)
    // ------------------------------------------------------------
    auto proj_result = project_vertices(mesh, camera);
    if (!proj_result) {
        return std::unexpected("Vertex projection failed: " + proj_result.error());
    }

    // ------------------------------------------------------------
    // Step 2: Compute screen-space bounding boxes for tetrahedra
    // ------------------------------------------------------------
    auto tile_result = compute_tile_assignments(mesh, width, height);
    if (!tile_result) {
        return std::unexpected("Tile assignment failed: " + tile_result.error());
    }

    // ------------------------------------------------------------
    // Step 3: Generate rays for all pixels on GPU
    // ------------------------------------------------------------
    core::Tensor ray_origins = core::Tensor::empty(
        {num_pixels, 3}, core::Device::CUDA, core::DataType::Float32);
    core::Tensor ray_directions = core::Tensor::empty(
        {num_pixels, 3}, core::Device::CUDA, core::DataType::Float32);

    // Ensure camera matrices are on GPU
    core::Tensor R_gpu = camera.R().to(core::Device::CUDA);
    core::Tensor T_gpu = camera.T().to(core::Device::CUDA);

    cuda::launch_generate_rays(
        ray_origins.ptr<float>(),
        ray_directions.ptr<float>(),
        R_gpu.ptr<float>(),
        T_gpu.ptr<float>(),
        camera.focal_x(),
        camera.focal_y(),
        camera.center_x(),
        camera.center_y(),
        width,
        height,
        nullptr);

    // ------------------------------------------------------------
    // Step 4: Get mesh data on GPU
    // ------------------------------------------------------------
    core::Tensor all_verts_gpu = mesh.get_all_vertices().to(core::Device::CUDA);
    core::Tensor tets_gpu = mesh.tetrahedra().to(core::Device::CUDA);

    // Validate data shapes
    if (!all_verts_gpu.is_valid() || all_verts_gpu.ndim() != 2) {
        return std::unexpected("Invalid vertices tensor");
    }
    if (!tets_gpu.is_valid() || tets_gpu.ndim() != 2) {
        return std::unexpected("Invalid tetrahedra tensor");
    }
    if (!tet_bounds_.is_valid() || tet_bounds_.ndim() != 2) {
        return std::unexpected("tet_bounds not properly initialized");
    }

    LOG_INFO("Render forward: {} vertices, {} tets, {}x{} image",
             all_verts_gpu.shape()[0], tets_gpu.shape()[0], width, height);

    // Validate tetrahedra indices are within vertex bounds
    // (This is a debug check - can be removed for performance)
    {
        core::Tensor tets_cpu = tets_gpu.to(core::Device::CPU);
        const int64_t* tet_data = tets_cpu.ptr<int64_t>();
        const int64_t max_valid_idx = static_cast<int64_t>(all_verts_gpu.shape()[0]) - 1;
        int64_t max_found = -1;
        int64_t min_found = std::numeric_limits<int64_t>::max();
        for (size_t t = 0; t < num_tets; ++t) {
            for (int v = 0; v < 4; ++v) {
                int64_t idx = tet_data[t * 4 + v];
                max_found = std::max(max_found, idx);
                min_found = std::min(min_found, idx);
                if (idx < 0 || idx > max_valid_idx) {
                    LOG_ERROR("Invalid tet vertex: tet[{}][{}] = {}, valid range [0, {}]",
                             t, v, idx, max_valid_idx);
                    return std::unexpected("Tetrahedra contain invalid vertex indices");
                }
            }
        }
        LOG_INFO("Tet vertex indices range: [{}, {}] (max valid: {})",
                 min_found, max_found, max_valid_idx);
    }

    // Ensure tet_bounds is on GPU (computed in compute_tile_assignments)
    core::Tensor tet_bounds_gpu = tet_bounds_.to(core::Device::CUDA);

    // ------------------------------------------------------------
    // Step 5: Allocate output tensors for intersections (all on GPU)
    // ------------------------------------------------------------
    core::Tensor intersection_ids = core::Tensor::full(
        {num_pixels, static_cast<size_t>(max_K)},
        static_cast<float>(-1),
        core::Device::CUDA,
        core::DataType::Int64);

    core::Tensor barycentric_coords = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K), 4},
        core::Device::CUDA,
        core::DataType::Float32);

    core::Tensor ray_depths = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K)},
        core::Device::CUDA,
        core::DataType::Float32);

    core::Tensor num_intersects = core::Tensor::zeros(
        {num_pixels},
        core::Device::CUDA,
        core::DataType::Int32);

    // ------------------------------------------------------------
    // Step 6: Tile-based ray-tet intersection (3DGS-style)
    // Much faster than brute-force: O(pixels × tets_per_tile) instead of O(pixels × tets)
    // ------------------------------------------------------------
    const int tile_size = config_.tile_size;
    const int tile_width = (width + tile_size - 1) / tile_size;
    const int tile_height = (height + tile_size - 1) / tile_size;
    const int num_tiles = tile_width * tile_height;

    LOG_INFO("Tile-based rasterization: {}x{} tiles ({}px each)", tile_width, tile_height, tile_size);

    // Step 6a: Compute centroid depth for each tet (for depth sorting)
    core::Tensor tet_depths = core::Tensor::zeros(
        {num_tets}, core::Device::CUDA, core::DataType::Float32);

    cuda::launch_compute_tet_depths(
        all_verts_gpu.ptr<float>(),
        tets_gpu.ptr<int64_t>(),
        R_gpu.ptr<float>(),
        T_gpu.ptr<float>(),
        tet_depths.ptr<float>(),
        num_tets,
        nullptr);

    // Step 6b: Count tiles touched by each tet
    core::Tensor tiles_per_tet = core::Tensor::zeros(
        {num_tets}, core::Device::CUDA, core::DataType::Int32);

    cuda::launch_count_tiles_per_tet(
        tet_bounds_gpu.ptr<float>(),
        tiles_per_tet.ptr<int32_t>(),
        num_tets,
        tile_size,
        tile_width,
        tile_height,
        width,
        height,
        nullptr);

    // Step 6c: Prefix sum to get offsets for pair generation
    core::Tensor cum_tiles = core::Tensor::zeros(
        {num_tets}, core::Device::CUDA, core::DataType::Int64);

    cuda::compute_prefix_sum_int32_to_int64(
        tiles_per_tet.ptr<int32_t>(),
        cum_tiles.ptr<int64_t>(),
        num_tets,
        nullptr);

    // Get total number of (tile, tet) pairs
    int64_t total_pairs = cuda::get_last_prefix_sum_value(
        cum_tiles.ptr<int64_t>(), num_tets, nullptr);

    LOG_INFO("Total tile-tet pairs: {} (avg {:.1f} tets/tile)",
             total_pairs, total_pairs > 0 ? static_cast<double>(total_pairs) / num_tiles : 0.0);

    // Initialize tile_ranges (even if no pairs, need valid ranges)
    core::Tensor tile_ranges = core::Tensor::zeros(
        {static_cast<size_t>(num_tiles), 2}, core::Device::CUDA, core::DataType::Int32);

    core::Tensor sorted_tet_ids = core::Tensor::zeros(
        {static_cast<size_t>(std::max(total_pairs, int64_t(1)))}, core::Device::CUDA, core::DataType::Int32);

    if (total_pairs > 0) {
        // Step 6d: Generate (tile_id, tet_idx) pairs for sorting
        core::Tensor isect_keys = core::Tensor::zeros(
            {static_cast<size_t>(total_pairs)}, core::Device::CUDA, core::DataType::Int64);
        core::Tensor tet_indices_unsorted = core::Tensor::zeros(
            {static_cast<size_t>(total_pairs)}, core::Device::CUDA, core::DataType::Int32);

        cuda::launch_generate_tet_tile_pairs(
            tet_bounds_gpu.ptr<float>(),
            tet_depths.ptr<float>(),
            cum_tiles.ptr<int64_t>(),
            isect_keys.ptr<int64_t>(),
            tet_indices_unsorted.ptr<int32_t>(),
            num_tets,
            tile_size,
            tile_width,
            tile_height,
            nullptr);

        // Step 6e: Radix sort by (tile_id, depth)
        core::Tensor sorted_keys = core::Tensor::zeros(
            {static_cast<size_t>(total_pairs)}, core::Device::CUDA, core::DataType::Int64);

        // Number of bits needed: tile_id uses ceil(log2(num_tiles)) + 32 for depth
        int tile_bits = static_cast<int>(ceil(log2(static_cast<double>(num_tiles + 1))));
        int num_key_bits = tile_bits + 32;

        cuda::radix_sort_tet_tiles(
            total_pairs,
            isect_keys.ptr<int64_t>(),
            tet_indices_unsorted.ptr<int32_t>(),
            sorted_keys.ptr<int64_t>(),
            sorted_tet_ids.ptr<int32_t>(),
            num_key_bits,
            nullptr);

        // Step 6f: Compute tile ranges from sorted keys
        cuda::launch_compute_tile_ranges(
            sorted_keys.ptr<int64_t>(),
            tile_ranges.ptr<int32_t>(),
            total_pairs,
            num_tiles,
            nullptr);
    }

    // Step 6g: Run tile-based intersection (fast!)
    LOG_INFO("Launching tile-based ray-tet intersection...");
    cuda::launch_ray_tet_intersection_with_tiles(
        ray_origins.ptr<float>(),
        ray_directions.ptr<float>(),
        all_verts_gpu.ptr<float>(),
        tets_gpu.ptr<int64_t>(),
        sorted_tet_ids.ptr<int32_t>(),
        tile_ranges.ptr<int32_t>(),
        intersection_ids.ptr<int64_t>(),
        ray_depths.ptr<float>(),
        barycentric_coords.ptr<float>(),
        num_intersects.ptr<int32_t>(),
        width,
        height,
        tile_size,
        tile_width,
        max_K,
        config_.near_clip,
        config_.far_clip,
        nullptr);

    // Debug: check for CUDA errors after intersection kernel
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after intersection kernel: {}", cudaGetErrorString(err));
            return std::unexpected("Ray-tet intersection kernel failed: " + std::string(cudaGetErrorString(err)));
        }
        LOG_INFO("Tile-based ray-tet intersection completed successfully");
    }

    // ------------------------------------------------------------
    // Step 7: Query neural network for per-tet colors/density
    // Uses hash grid + MLP (iNGP-style) for high-capacity representation
    // ------------------------------------------------------------

    // Step 7a: Compute tetrahedron centroids for network queries
    core::Tensor tet_centroids = core::Tensor::zeros(
        {num_tets, 3}, core::Device::CUDA, core::DataType::Float32);

    cuda::launch_compute_tet_centroids(
        all_verts_gpu.ptr<float>(),
        tets_gpu.ptr<int64_t>(),
        tet_centroids.ptr<float>(),
        num_tets,
        nullptr);

    // Step 7b: Query neural network for per-tet features
    // Create view directions (use camera position to tet centroid direction)
    // For now, use zeros - view-dependent effects will be added later
    core::Tensor view_dirs = core::Tensor::zeros(
        {num_tets, 3}, core::Device::CUDA, core::DataType::Float32);

    // Network integration enabled - kernel now supports both modes:
    // - When use_network_colors=true: use network RGB directly (sigmoid-activated)
    // - When use_network_colors=false: use per-tet linear model with softplus
    bool use_network = true;  // Use neural network for high-capacity representation

    // Determine which parameters to use (network or per-tet fallback)
    const float* tet_density_ptr = nullptr;
    const float* tet_base_color_ptr = nullptr;
    const float* tet_gradient_ptr = nullptr;

    // Tensors to hold network predictions (must stay in scope)
    core::Tensor network_density;
    core::Tensor network_base_color;
    core::Tensor network_gradient;
    bool used_network = false;

    if (use_network) {
        // Query network with centroids to get RGB and density
        auto feature_result = features.forward_with_cache(tet_centroids, view_dirs);
        if (feature_result) {
            LOG_INFO("Neural network predicted colors for {} tetrahedra", num_tets);
            used_network = true;

            // Network RGB -> base_color
            network_base_color = feature_result->rgb;

            // Network density (if available) or use per-tet density
            if (feature_result->density.is_valid()) {
                network_density = feature_result->density;
                tet_density_ptr = network_density.ptr<float>();
            } else {
                tet_density_ptr = mesh.density().ptr<float>();
            }

            tet_base_color_ptr = network_base_color.ptr<float>();

            // Use zeros for gradient (no spatial variation within tet for now)
            network_gradient = core::Tensor::zeros(
                {num_tets, 3}, core::Device::CUDA, core::DataType::Float32);
            tet_gradient_ptr = network_gradient.ptr<float>();
        }
    }

    if (!used_network) {
        // Use per-tet parameters (default path)
        tet_density_ptr = mesh.density().ptr<float>();
        tet_base_color_ptr = mesh.base_color().ptr<float>();
        tet_gradient_ptr = mesh.gradient().ptr<float>();
    }

    // Step 7c: Compute RGB and alpha for each intersection
    core::Tensor sample_rgb = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K), 3},
        core::Device::CUDA,
        core::DataType::Float32);

    core::Tensor sample_alpha = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K)},
        core::Device::CUDA,
        core::DataType::Float32);

    LOG_INFO("Launching compute_intersection_colors kernel (use_network={})...", used_network);
    cuda::launch_compute_intersection_colors(
        all_verts_gpu.ptr<float>(),
        tets_gpu.ptr<int64_t>(),
        intersection_ids.ptr<int64_t>(),
        barycentric_coords.ptr<float>(),
        ray_depths.ptr<float>(),
        num_intersects.ptr<int32_t>(),
        tet_density_ptr,
        tet_base_color_ptr,
        tet_gradient_ptr,
        sample_rgb.ptr<float>(),
        sample_alpha.ptr<float>(),
        width,
        height,
        max_K,
        used_network,  // Skip softplus when using network colors
        nullptr);
    // Debug: check for CUDA errors after compute colors kernel
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after compute_intersection_colors kernel: {}", cudaGetErrorString(err));
            return std::unexpected("compute_intersection_colors kernel failed: " + std::string(cudaGetErrorString(err)));
        }
        LOG_INFO("compute_intersection_colors kernel completed successfully");
    }

    // ------------------------------------------------------------
    // Step 8: Alpha blending on GPU
    // ------------------------------------------------------------
    core::Tensor rgb = core::Tensor::zeros(
        {static_cast<size_t>(height), static_cast<size_t>(width), 3},
        core::Device::CUDA,
        core::DataType::Float32);

    core::Tensor alpha = core::Tensor::zeros(
        {static_cast<size_t>(height), static_cast<size_t>(width), 1},
        core::Device::CUDA,
        core::DataType::Float32);

    // Ensure background is on GPU
    core::Tensor bg_gpu = background.to(core::Device::CUDA);
    bool bg_is_image = (bg_gpu.ndim() == 3);

    LOG_INFO("Launching alpha_blend kernel...");
    cuda::launch_alpha_blend(
        sample_rgb.ptr<float>(),
        sample_alpha.ptr<float>(),
        ray_depths.ptr<float>(),
        num_intersects.ptr<int32_t>(),
        bg_gpu.ptr<float>(),
        rgb.ptr<float>(),
        alpha.ptr<float>(),
        width,
        height,
        max_K,
        config_.alpha_threshold,
        bg_is_image,
        nullptr);
    // Debug: check for CUDA errors after alpha blend kernel
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after alpha_blend kernel: {}", cudaGetErrorString(err));
            return std::unexpected("alpha_blend kernel failed: " + std::string(cudaGetErrorString(err)));
        }
        LOG_INFO("alpha_blend kernel completed successfully");
    }

    // ------------------------------------------------------------
    // Step 9: Compute depth buffer (first intersection depth)
    // ------------------------------------------------------------
    core::Tensor depth_tensor = core::Tensor::zeros(
        {static_cast<size_t>(height), static_cast<size_t>(width), 1},
        core::Device::CUDA,
        core::DataType::Float32);

    if (config_.compute_depth) {
        cuda::launch_compute_depth_buffer(
            ray_depths.ptr<float>(),
            num_intersects.ptr<int32_t>(),
            depth_tensor.ptr<float>(),
            width,
            height,
            max_K,
            nullptr);
    }

    // ------------------------------------------------------------
    // Step 10: Reshape output tensors to [H, W, ...] format
    // ------------------------------------------------------------
    // Reshape intersection_ids from [H*W, max_K] to [H, W, max_K]
    core::Tensor intersection_ids_reshaped = intersection_ids.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(max_K)});

    // Reshape barycentric_coords from [H*W, max_K, 4] to [H, W, max_K, 4]
    core::Tensor barycentric_coords_reshaped = barycentric_coords.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(max_K), 4});

    // Reshape ray_depths from [H*W, max_K] to [H, W, max_K]
    core::Tensor ray_depths_reshaped = ray_depths.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(max_K)});

    // Reshape num_intersects from [H*W] to [H, W]
    core::Tensor num_intersects_reshaped = num_intersects.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width)});

    // Reshape sample_rgb from [H*W, max_K, 3] to [H, W, max_K, 3]
    core::Tensor sample_rgb_reshaped = sample_rgb.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width),
         static_cast<size_t>(max_K), 3});

    // Reshape sample_alpha from [H*W, max_K] to [H, W, max_K]
    core::Tensor sample_alpha_reshaped = sample_alpha.reshape(
        {static_cast<size_t>(height), static_cast<size_t>(width),
         static_cast<size_t>(max_K)});

    return RenderForwardResult{
        std::move(rgb),
        std::move(alpha),
        std::move(depth_tensor),
        std::move(num_intersects_reshaped),
        std::move(intersection_ids_reshaped),
        std::move(barycentric_coords_reshaped),
        std::move(ray_depths_reshaped),
        std::move(sample_rgb_reshaped),
        std::move(sample_alpha_reshaped),
        std::move(tet_centroids),
        std::move(view_dirs),
        used_network
    };
}

std::expected<RenderForwardResult, std::string> TetraRenderer::forward_with_cache(
    const TetraMesh& mesh,
    TetraFeatures& features,
    const core::Camera& camera,
    const core::Tensor& background,
    const RenderForwardResult& cached_intersections) {

    // For cached rendering, skip intersection computation
    // and directly compute colors from stored barycentric coords
    // This is a stub - full implementation would be needed

    return forward(mesh, features, camera, background);
}

// ------------------------------
// BACKWARD PASS
// ------------------------------

std::expected<RenderBackwardResult, std::string> TetraRenderer::backward(
    const TetraMesh& mesh,
    TetraFeatures& features,
    const core::Camera& camera,
    const RenderForwardResult& forward_result,
    const core::Tensor& grad_rgb,
    const core::Tensor& grad_alpha) {

    LOG_INFO("Backward pass starting...");

    const int width = camera.image_width();
    const int height = camera.image_height();
    const int max_K = config_.max_intersections_per_pixel;
    const size_t num_verts = mesh.num_vertices();
    const size_t num_pixels = static_cast<size_t>(width * height);

    // Convert grad_rgb from CHW [C, H, W] to HWC [H, W, C] for gradient propagation
    core::Tensor grad_rgb_hwc = grad_rgb.to(core::Device::CUDA);
    if (grad_rgb_hwc.ndim() == 3 && grad_rgb_hwc.shape()[0] <= 4) {
        // CHW format [C, H, W] -> HWC [H, W, C]
        LOG_INFO("Converting grad_rgb from CHW [{}, {}, {}] to HWC",
                 grad_rgb_hwc.shape()[0], grad_rgb_hwc.shape()[1], grad_rgb_hwc.shape()[2]);
        grad_rgb_hwc = grad_rgb_hwc.permute({1, 2, 0}).contiguous();
        LOG_INFO("grad_rgb_hwc shape: [{}, {}, {}]",
                 grad_rgb_hwc.shape()[0], grad_rgb_hwc.shape()[1], grad_rgb_hwc.shape()[2]);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error before backward kernel: {}", cudaGetErrorString(err));
        return std::unexpected(std::string("CUDA error before backward: ") + cudaGetErrorString(err));
    }

    // ------------------------------------------------------------
    // Step 1: Flatten forward_result tensors for GPU kernel access
    // ------------------------------------------------------------
    core::Tensor num_intersects_flat = forward_result.num_intersects.reshape(
        {num_pixels}).contiguous();
    core::Tensor bary_flat = forward_result.barycentric_coords.reshape(
        {num_pixels * static_cast<size_t>(max_K), 4}).contiguous();
    core::Tensor tet_ids_flat = forward_result.intersection_ids.reshape(
        {num_pixels * static_cast<size_t>(max_K)}).contiguous();
    core::Tensor ray_depths_flat = forward_result.ray_depths.reshape(
        {num_pixels, static_cast<size_t>(max_K)}).contiguous();

    // Flatten stored sample_rgb and sample_alpha from [H, W, max_K, ...] to [H*W, max_K, ...]
    core::Tensor sample_rgb_flat = forward_result.sample_rgb.reshape(
        {num_pixels, static_cast<size_t>(max_K), 3}).contiguous();
    core::Tensor sample_alpha_flat = forward_result.sample_alpha.reshape(
        {num_pixels, static_cast<size_t>(max_K)}).contiguous();

    // ------------------------------------------------------------
    // Step 2: Run alpha_blend_backward to compute per-sample gradients
    // This kernel traverses samples in reverse order (back-to-front)
    // and computes d_loss/d_rgb_sample and d_loss/d_alpha_sample
    // ------------------------------------------------------------
    core::Tensor grad_sample_rgb = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K), 3},
        core::Device::CUDA,
        core::DataType::Float32);

    core::Tensor grad_sample_alpha = core::Tensor::zeros(
        {num_pixels, static_cast<size_t>(max_K)},
        core::Device::CUDA,
        core::DataType::Float32);

    // Prepare grad_alpha for kernel (ensure on GPU and correct shape)
    core::Tensor grad_alpha_gpu = grad_alpha.to(core::Device::CUDA);
    if (grad_alpha_gpu.ndim() == 3 && grad_alpha_gpu.shape()[0] <= 4) {
        // CHW format [C, H, W] -> HWC [H, W, C]
        grad_alpha_gpu = grad_alpha_gpu.permute({1, 2, 0}).contiguous();
    }

    LOG_INFO("Launching alpha_blend_backward kernel...");
    cuda::launch_alpha_blend_backward(
        sample_rgb_flat.ptr<float>(),
        sample_alpha_flat.ptr<float>(),
        ray_depths_flat.ptr<float>(),
        num_intersects_flat.ptr<int32_t>(),
        grad_rgb_hwc.ptr<float>(),
        grad_alpha_gpu.ptr<float>(),
        grad_sample_rgb.ptr<float>(),
        grad_sample_alpha.ptr<float>(),
        width,
        height,
        max_K,
        config_.alpha_threshold,
        nullptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error after alpha_blend_backward kernel: {}", cudaGetErrorString(err));
        return std::unexpected(std::string("CUDA error in alpha_blend_backward: ") + cudaGetErrorString(err));
    }
    LOG_INFO("alpha_blend_backward kernel completed successfully");

    // ------------------------------------------------------------
    // Step 2b: Compute gradients for per-tet color parameters
    // ------------------------------------------------------------
    const size_t num_tets = mesh.num_tetrahedra();

    // Allocate gradient tensors for per-tet parameters
    core::Tensor grad_density = core::Tensor::zeros(
        {num_tets}, core::Device::CUDA, core::DataType::Float32);
    core::Tensor grad_base_color = core::Tensor::zeros(
        {num_tets, 3}, core::Device::CUDA, core::DataType::Float32);
    core::Tensor grad_gradient = core::Tensor::zeros(
        {num_tets, 3}, core::Device::CUDA, core::DataType::Float32);

    // Get mesh data on GPU
    core::Tensor vertices_gpu = mesh.get_all_vertices().to(core::Device::CUDA);
    core::Tensor tets_gpu = mesh.tetrahedra().to(core::Device::CUDA);

    LOG_INFO("Launching compute_intersection_colors_backward kernel...");
    const size_t num_vertices = mesh.num_vertices();
    cuda::launch_compute_intersection_colors_backward(
        vertices_gpu.ptr<float>(),
        tets_gpu.ptr<int64_t>(),
        tet_ids_flat.ptr<int64_t>(),
        bary_flat.ptr<float>(),
        ray_depths_flat.ptr<float>(),
        num_intersects_flat.ptr<int32_t>(),
        mesh.density().ptr<float>(),
        mesh.base_color().ptr<float>(),
        mesh.gradient().ptr<float>(),
        grad_sample_rgb.ptr<float>(),
        grad_sample_alpha.ptr<float>(),
        grad_density.ptr<float>(),
        grad_base_color.ptr<float>(),
        grad_gradient.ptr<float>(),
        width,
        height,
        max_K,
        static_cast<int64_t>(num_tets),
        static_cast<int64_t>(num_vertices),
        nullptr);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error after compute_intersection_colors_backward kernel: {}", cudaGetErrorString(err));
        return std::unexpected(std::string("CUDA error in compute_intersection_colors_backward: ") + cudaGetErrorString(err));
    }
    LOG_INFO("compute_intersection_colors_backward kernel completed successfully");

    // ------------------------------------------------------------
    // Step 3: Compute prefix sum for query collection
    // ------------------------------------------------------------
    core::Tensor prefix_sum = core::Tensor::empty(
        {num_pixels}, core::Device::CUDA, core::DataType::Int32);

    cuda::launch_prefix_sum(
        num_intersects_flat.ptr<int32_t>(),
        prefix_sum.ptr<int32_t>(),
        num_pixels,
        nullptr);

    // Get total queries from prefix sum
    core::Tensor prefix_sum_cpu = prefix_sum.to(core::Device::CPU);
    size_t total_queries = static_cast<size_t>(prefix_sum_cpu.ptr<int32_t>()[num_pixels - 1]);

    // ------------------------------------------------------------
    // Step 4: Collect query positions and propagate gradients
    // ------------------------------------------------------------
    core::Tensor query_positions;
    core::Tensor query_directions;
    core::Tensor grad_features;

    if (total_queries > 0) {
        // Allocate output tensors on GPU
        // grad_features has 4 channels: 3 for RGB gradient + 1 for alpha gradient
        query_positions = core::Tensor::empty(
            {total_queries, 3}, core::Device::CUDA, core::DataType::Float32);
        query_directions = core::Tensor::empty(
            {total_queries, 3}, core::Device::CUDA, core::DataType::Float32);
        grad_features = core::Tensor::empty(
            {total_queries, 4}, core::Device::CUDA, core::DataType::Float32);

        // Get camera position on GPU
        core::Tensor cam_pos = camera.T().to(core::Device::CUDA).contiguous();

        // Reconstruct intersection positions and directions using GPU kernel
        cuda::launch_reconstruct_intersections(
            mesh.vertices().ptr<float>(),
            mesh.tetrahedra().ptr<int64_t>(),
            bary_flat.ptr<float>(),
            tet_ids_flat.ptr<int64_t>(),
            num_intersects_flat.ptr<int32_t>(),
            cam_pos.ptr<float>(),
            query_positions.ptr<float>(),
            query_directions.ptr<float>(),
            prefix_sum.ptr<int32_t>(),
            width, height, max_K,
            nullptr);

        // Propagate per-sample gradients (from alpha_blend_backward) to features
        // The grad_sample_rgb [H*W, max_K, 3] and grad_sample_alpha [H*W, max_K]
        // need to be collected into grad_features [N_query, 4] using the prefix sum
        cuda::launch_propagate_gradients(
            grad_sample_rgb.ptr<float>(),
            num_intersects_flat.ptr<int32_t>(),
            prefix_sum.ptr<int32_t>(),
            grad_features.ptr<float>(),
            width, height, max_K,
            nullptr);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in backward gradient propagation: {}", cudaGetErrorString(err));
            return std::unexpected(std::string("CUDA error in gradient propagation: ") + cudaGetErrorString(err));
        }
    }

    // ------------------------------------------------------------
    // Step 5: Compute vertex gradients through barycentric interpolation
    // ------------------------------------------------------------
    // Gradient flows from grad_features (RGB gradient per sample)
    // through barycentric interpolation to the 4 vertices of each tet.
    //
    // For barycentric interpolation:
    //   position = sum(bary[i] * vertex[i] for i in 0..3)
    // Therefore:
    //   d_position/d_vertex[i] = bary[i] * I (identity matrix)
    //   d_loss/d_vertex[i] = d_loss/d_position * bary[i]
    // ------------------------------------------------------------
    core::Tensor grad_vertices = core::Tensor::zeros(
        {num_verts, 3}, core::Device::CUDA, core::DataType::Float32);

    if (total_queries > 0 && grad_features.is_valid()) {
        LOG_INFO("Launching vertex_gradients kernel for {} queries, {} vertices...",
                 total_queries, num_verts);

        cuda::launch_vertex_gradients(
            grad_features.ptr<float>(),
            bary_flat.ptr<float>(),
            tet_ids_flat.ptr<int64_t>(),
            tets_gpu.ptr<int64_t>(),
            num_intersects_flat.ptr<int32_t>(),
            prefix_sum.ptr<int32_t>(),
            grad_vertices.ptr<float>(),
            width,
            height,
            max_K,
            num_verts,
            nullptr);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in vertex_gradients kernel: {}", cudaGetErrorString(err));
            return std::unexpected(std::string("CUDA error in vertex_gradients: ") + cudaGetErrorString(err));
        }
        LOG_INFO("vertex_gradients kernel completed successfully");
    }

    return RenderBackwardResult{
        std::move(grad_vertices),
        std::move(grad_features),
        std::move(query_positions),
        std::move(query_directions),
        std::move(grad_density),
        std::move(grad_base_color),
        std::move(grad_gradient)
    };
}

// ------------------------------
// INTERNAL METHODS
// ------------------------------

std::expected<void, std::string> TetraRenderer::project_vertices(
    const TetraMesh& mesh,
    const core::Camera& camera) {

    core::Tensor verts = mesh.get_all_vertices();
    const size_t num_verts = static_cast<size_t>(verts.shape()[0]);

    // Ensure vertices are on GPU
    core::Tensor verts_gpu = verts.to(core::Device::CUDA);

    // Allocate projected vertices [V, 4] on GPU
    projected_verts_ = core::Tensor::empty(
        {num_verts, 4},
        core::Device::CUDA, core::DataType::Float32);

    // Get camera matrices on GPU
    core::Tensor R_gpu = camera.R().to(core::Device::CUDA);
    core::Tensor T_gpu = camera.T().to(core::Device::CUDA);

    // Create intrinsics tensor [fx, fy, cx, cy]
    core::Tensor intrinsics_cpu = core::Tensor::empty(
        {4}, core::Device::CPU, core::DataType::Float32);
    float* intr_data = intrinsics_cpu.ptr<float>();
    intr_data[0] = camera.focal_x();
    intr_data[1] = camera.focal_y();
    intr_data[2] = camera.center_x();
    intr_data[3] = camera.center_y();
    core::Tensor intrinsics_gpu = intrinsics_cpu.to(core::Device::CUDA);

    // Launch GPU kernel for vertex projection
    cuda::launch_project_vertices(
        verts_gpu.ptr<float>(),
        R_gpu.ptr<float>(),
        T_gpu.ptr<float>(),
        intrinsics_gpu.ptr<float>(),
        projected_verts_.ptr<float>(),
        num_verts,
        config_.near_clip,
        nullptr);

    return {};
}

std::expected<void, std::string> TetraRenderer::compute_tile_assignments(
    const TetraMesh& mesh,
    int width, int height) {

    // Compute screen-space bounds for each tetrahedron on GPU
    const size_t num_tets = mesh.num_tetrahedra();

    // Allocate bounds tensor on GPU
    tet_bounds_ = core::Tensor::empty(
        {num_tets, 4},
        core::Device::CUDA, core::DataType::Float32);

    // Ensure tetrahedra are on GPU
    core::Tensor tets_gpu = mesh.tetrahedra().to(core::Device::CUDA);

    // Launch GPU kernel for bounding box computation
    cuda::launch_compute_tet_bounds(
        tets_gpu.ptr<int64_t>(),
        projected_verts_.ptr<float>(),
        tet_bounds_.ptr<float>(),
        num_tets,
        nullptr);

    return {};
}

// ------------------------------
// UTILITIES
// ------------------------------

core::Tensor TetraRenderer::compute_visibility_scores(
    const TetraMesh& mesh,
    const std::vector<core::Camera>& cameras) {

    const size_t num_tets = mesh.num_tetrahedra();

    // Initialize scores to zero
    core::Tensor scores = core::Tensor::zeros(
        {num_tets},
        core::Device::CPU, core::DataType::Float32);

    // For each camera, render and accumulate visibility
    // This is a stub - full implementation would trace rays

    return scores.to(core::Device::CUDA);
}

} // namespace lfs::tetra
