/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <expected>
#include <string>
#include <tuple>

namespace lfs::tetra {

/**
 * @brief Delaunay triangulation utilities for tetrahedral mesh construction
 *
 * Provides both CPU (CGAL-based) and GPU (gDel3D-style) implementations
 * for computing 3D Delaunay triangulations.
 */
namespace delaunay {

    /**
     * @brief Result of Delaunay triangulation
     */
    struct TriangulationResult {
        core::Tensor tetrahedra;    // [T, 4] int64 - tetrahedron vertex indices
        core::Tensor neighbors;     // [T, 4] int64 - neighboring tetrahedra (-1 for boundary)
        size_t num_tetrahedra;
    };

    /**
     * @brief Compute 3D Delaunay triangulation on CPU using CGAL
     *
     * @param vertices [N, 3] float32 - vertex positions
     * @return TriangulationResult on success, error string on failure
     */
    std::expected<TriangulationResult, std::string> triangulate_cpu(
        const core::Tensor& vertices);

    /**
     * @brief Compute 3D Delaunay triangulation on GPU
     *
     * Uses a gDel3D-style algorithm for fast GPU triangulation.
     * Falls back to CPU if GPU triangulation fails.
     *
     * @param vertices [N, 3] float32 - vertex positions
     * @param stream CUDA stream for async execution
     * @return TriangulationResult on success, error string on failure
     */
    std::expected<TriangulationResult, std::string> triangulate_gpu(
        const core::Tensor& vertices,
        void* stream = nullptr);

    /**
     * @brief Create exterior shell vertices for bounding the scene
     *
     * Generates vertices on a sphere/box enclosing the input points.
     * These form the exterior boundary for the Delaunay triangulation.
     *
     * @param vertices [N, 3] float32 - interior vertex positions
     * @param expansion_factor Scale factor for shell size (default 1.5)
     * @param num_shell_points Number of shell points (default 8 for cube corners)
     * @return Shell vertices [M, 3] float32
     */
    core::Tensor create_exterior_shell(
        const core::Tensor& vertices,
        float expansion_factor = 1.5f,
        int num_shell_points = 8);

    /**
     * @brief Validate tetrahedral mesh quality
     *
     * Checks for degenerate tetrahedra, proper orientation, etc.
     *
     * @param vertices [N, 3] float32 - all vertices
     * @param tetrahedra [T, 4] int64 - tetrahedron indices
     * @return Empty on valid mesh, error description otherwise
     */
    std::expected<void, std::string> validate_mesh(
        const core::Tensor& vertices,
        const core::Tensor& tetrahedra);

} // namespace delaunay

} // namespace lfs::tetra
