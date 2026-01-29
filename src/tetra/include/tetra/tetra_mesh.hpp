/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include "core/tensor.hpp"

#include <expected>
#include <filesystem>
#include <string>

namespace lfs::tetra {

/**
 * @brief Per-tetrahedron features for PLY export (radiance_meshes format)
 *
 * Contains computed features for each tetrahedron:
 * - density: [T] scalar density values
 * - gradient: [T, 3] normalized gradient vectors (grd_x, grd_y, grd_z)
 * - sh_dc: [T, 3] DC spherical harmonics (sh_0_r, sh_0_g, sh_0_b)
 * - sh_rest: [T, sh_dim, 3] higher-order SH coefficients (sh_1..sh_15)
 *
 * This matches the Python save2ply() format from radiance_meshes.
 */
struct TetraPlyFeatures {
    core::Tensor density;   // [T] float32
    core::Tensor gradient;  // [T, 3] float32
    core::Tensor sh_dc;     // [T, 3] float32 (DC color converted to SH)
    core::Tensor sh_rest;   // [T, sh_dim, 3] float32 (sh_dim typically 15)
};

/**
 * @brief Core data structure for tetrahedral mesh representation
 *
 * Contains the fundamental attributes of a tetrahedral mesh scene:
 * - Interior vertices (optimizable positions)
 * - Exterior shell vertices (fixed boundary)
 * - Tetrahedra indices (connectivity)
 *
 * This follows the Radiance Meshes paper architecture where:
 * - Interior vertices are optimized during training
 * - Exterior vertices form a bounding shell
 * - Delaunay triangulation maintains mesh connectivity
 *
 * Note: Gradients are managed by AdamOptimizer, not TetraMesh.
 */
class TetraMesh {
public:
    TetraMesh() = default;
    ~TetraMesh();

    // Delete copy operations
    TetraMesh(const TetraMesh&) = delete;
    TetraMesh& operator=(const TetraMesh&) = delete;

    // Custom move operations
    TetraMesh(TetraMesh&& other) noexcept;
    TetraMesh& operator=(TetraMesh&& other) noexcept;

    // Constructor
    TetraMesh(core::Tensor vertices,
              core::Tensor ext_vertices,
              core::Tensor tetrahedra,
              float scene_scale);

    // ------------------------------
    // FACTORY METHODS
    // ------------------------------

    /**
     * @brief Create TetraMesh from a PointCloud via Delaunay triangulation
     * @param point_cloud Source point cloud (positions)
     * @param scene_scale Scale factor for the scene
     * @param shell_expansion Factor to expand bounding shell (default 1.5)
     * @return TetraMesh on success, error string on failure
     */
    static std::expected<TetraMesh, std::string> from_point_cloud(
        const core::PointCloud& point_cloud,
        float scene_scale,
        float shell_expansion = 1.5f);

    // ------------------------------
    // COMPUTED GETTERS
    // ------------------------------

    /**
     * @brief Get all vertices (interior + exterior concatenated)
     * @return Tensor [V_int + V_ext, 3] of all vertex positions
     */
    core::Tensor get_all_vertices() const;

    // ------------------------------
    // SIMPLE INLINE GETTERS
    // ------------------------------

    [[nodiscard]] size_t num_interior_vertices() const {
        return vertices_.is_valid() ? static_cast<size_t>(vertices_.shape()[0]) : 0;
    }

    [[nodiscard]] size_t num_exterior_vertices() const {
        return ext_vertices_.is_valid() ? static_cast<size_t>(ext_vertices_.shape()[0]) : 0;
    }

    [[nodiscard]] size_t num_vertices() const {
        return num_interior_vertices() + num_exterior_vertices();
    }

    [[nodiscard]] size_t num_tetrahedra() const {
        return tetrahedra_.is_valid() ? static_cast<size_t>(tetrahedra_.shape()[0]) : 0;
    }

    [[nodiscard]] float get_scene_scale() const { return scene_scale_; }

    // ------------------------------
    // RAW TENSOR ACCESS (for optimization)
    // ------------------------------

    [[nodiscard]] core::Tensor& vertices() { return vertices_; }
    [[nodiscard]] const core::Tensor& vertices() const { return vertices_; }
    [[nodiscard]] core::Tensor& ext_vertices() { return ext_vertices_; }
    [[nodiscard]] const core::Tensor& ext_vertices() const { return ext_vertices_; }
    [[nodiscard]] core::Tensor& tetrahedra() { return tetrahedra_; }
    [[nodiscard]] const core::Tensor& tetrahedra() const { return tetrahedra_; }

    // Per-tetrahedron learnable parameters
    [[nodiscard]] core::Tensor& density() { return density_; }
    [[nodiscard]] const core::Tensor& density() const { return density_; }
    [[nodiscard]] core::Tensor& base_color() { return base_color_; }
    [[nodiscard]] const core::Tensor& base_color() const { return base_color_; }
    [[nodiscard]] core::Tensor& gradient() { return gradient_; }
    [[nodiscard]] const core::Tensor& gradient() const { return gradient_; }

    // ------------------------------
    // VERTEX MODIFICATION
    // ------------------------------

    /**
     * @brief Add new interior vertices to the mesh
     *
     * Used during densification to add vertices at high-gradient regions.
     * The triangulation should be updated after adding vertices.
     *
     * @param new_vertices [N, 3] New vertex positions to add
     * @return Error string on failure
     */
    std::expected<void, std::string> add_vertices(const core::Tensor& new_vertices);

    // ------------------------------
    // TRIANGULATION UPDATE
    // ------------------------------

    /**
     * @brief Recompute Delaunay triangulation from current vertices
     *
     * Called periodically during training to maintain valid tetrahedralization.
     * Only interior vertices are used; exterior shell remains fixed.
     *
     * @return Error string on failure
     */
    std::expected<void, std::string> update_triangulation();

    // ------------------------------
    // MESH EXTRACTION
    // ------------------------------

    /**
     * @brief Extract surface triangle mesh from tetrahedra
     *
     * Computes boundary triangles from the tetrahedral mesh.
     * Used for mesh export and collision detection.
     *
     * @param visibility_threshold Minimum visibility score to include tet (0-1)
     * @return Tuple of (vertices [V,3], triangles [T,3], colors [V,3])
     */
    std::expected<std::tuple<core::Tensor, core::Tensor, core::Tensor>, std::string>
    extract_surface_mesh(float visibility_threshold = 0.1f) const;

    // ------------------------------
    // SERIALIZATION
    // ------------------------------

    void serialize(std::ostream& os) const;
    void deserialize(std::istream& is);

    /**
     * @brief Save to PLY file (tetrahedral format, geometry only)
     */
    void save_ply(const std::filesystem::path& path) const;

    /**
     * @brief Save to PLY file with full model state (radiance_meshes format)
     *
     * Exports the complete model state matching the Python save2ply() format:
     * - Vertex element: x, y, z positions
     * - Tetrahedron element:
     *   - indices: 4 vertex indices
     *   - s: density (float)
     *   - grd_x, grd_y, grd_z: gradient (float)
     *   - sh_0_r, sh_0_g, sh_0_b: DC color as SH (float)
     *   - sh_1_r..sh_15_r/g/b: higher order SH coefficients (float)
     *
     * @param path Output PLY file path
     * @param features Per-tetrahedron computed features
     */
    void save_ply(const std::filesystem::path& path,
                  const TetraPlyFeatures& features) const;

    /**
     * @brief Load from PLY file (tetrahedral format)
     */
    static std::expected<TetraMesh, std::string> load_ply(const std::filesystem::path& path);

    /**
     * @brief Export as triangle mesh PLY (for viewers/engines)
     */
    void export_triangle_mesh(const std::filesystem::path& path,
                              float visibility_threshold = 0.1f) const;

private:
    float scene_scale_ = 0.f;

    // Interior vertices [V_int, 3] - optimizable
    core::Tensor vertices_;

    // Exterior shell vertices [V_ext, 3] - fixed boundary
    core::Tensor ext_vertices_;

    // Tetrahedra indices [T, 4] - connectivity (int64)
    core::Tensor tetrahedra_;

    // Per-tetrahedron visibility scores [T] - computed during training
    mutable core::Tensor visibility_scores_;

    // Per-tetrahedron learnable parameters
    core::Tensor density_;     // [T] float32 - opacity control
    core::Tensor base_color_;  // [T, 3] float32 - DC color
    core::Tensor gradient_;    // [T, 3] float32 - spatial color variation

    /**
     * @brief Initialize per-tetrahedron learnable parameters
     *
     * Called after triangulation to create learnable parameters for each tet:
     * - density: initialized to -2.0 (log scale, small positive opacity)
     * - base_color: initialized to 0.5 (gray)
     * - gradient: initialized to 0.0 (no spatial variation)
     */
    void init_per_tet_params();
};

} // namespace lfs::tetra
