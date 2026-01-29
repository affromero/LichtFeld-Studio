/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_delaunay.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#ifdef LFS_HAS_GPU_DELAUNAY
#include "gDel3D/GpuDelaunay.h"
#include "gDel3D/CommonTypes.h"
#endif

namespace lfs::tetra::delaunay {

namespace {

/**
 * @brief Simple incremental Delaunay implementation
 *
 * This is a basic implementation without CGAL dependency.
 * For production use, consider integrating CGAL or TetGen.
 */
class IncrementalDelaunay {
public:
    struct Point {
        float x, y, z;
        int64_t original_index;

        Point() : x(0), y(0), z(0), original_index(-1) {}
        Point(float x_, float y_, float z_, int64_t idx)
            : x(x_), y(y_), z(z_), original_index(idx) {}
    };

    struct Tetrahedron {
        int64_t v[4];      // Vertex indices
        int64_t adj[4];    // Adjacent tetrahedra (-1 for boundary)
        bool valid;

        Tetrahedron() : valid(true) {
            for (int i = 0; i < 4; ++i) {
                v[i] = -1;
                adj[i] = -1;
            }
        }
    };

    explicit IncrementalDelaunay(std::vector<Point> points)
        : points_(std::move(points)) {}

    std::expected<TriangulationResult, std::string> triangulate() {
        if (points_.size() < 4) {
            return std::unexpected("Need at least 4 points for tetrahedralization");
        }

        // Create super-tetrahedron that contains all points
        createSuperTetrahedron();

        // Insert points one by one
        for (size_t i = 0; i < points_.size(); ++i) {
            if (!insertPoint(static_cast<int64_t>(i))) {
                return std::unexpected("Failed to insert point " + std::to_string(i));
            }
        }

        // Remove super-tetrahedron vertices and associated tetrahedra
        removeSuperTetrahedron();

        // Build result
        return buildResult();
    }

private:
    std::vector<Point> points_;
    std::vector<Tetrahedron> tetrahedra_;
    std::vector<Point> super_vertices_;
    int64_t super_start_idx_;

    void createSuperTetrahedron() {
        // Find bounding box
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();

        for (const auto& p : points_) {
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            min_z = std::min(min_z, p.z);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
            max_z = std::max(max_z, p.z);
        }

        // Expand bounding box
        float dx = (max_x - min_x) * 10.0f + 1.0f;
        float dy = (max_y - min_y) * 10.0f + 1.0f;
        float dz = (max_z - min_z) * 10.0f + 1.0f;
        float cx = (min_x + max_x) * 0.5f;
        float cy = (min_y + max_y) * 0.5f;
        float cz = (min_z + max_z) * 0.5f;

        // Create super-tetrahedron vertices
        super_start_idx_ = static_cast<int64_t>(points_.size());

        // Super-tetrahedron vertices (large enough to contain all points)
        super_vertices_ = {
            Point(cx, cy + dy * 3, cz, super_start_idx_),
            Point(cx - dx * 2, cy - dy, cz - dz, super_start_idx_ + 1),
            Point(cx + dx * 2, cy - dy, cz - dz, super_start_idx_ + 2),
            Point(cx, cy - dy, cz + dz * 2, super_start_idx_ + 3)
        };

        // Add super vertices to points
        for (const auto& sv : super_vertices_) {
            points_.push_back(sv);
        }

        // Create initial tetrahedron
        Tetrahedron t;
        t.v[0] = super_start_idx_;
        t.v[1] = super_start_idx_ + 1;
        t.v[2] = super_start_idx_ + 2;
        t.v[3] = super_start_idx_ + 3;
        tetrahedra_.push_back(t);
    }

    bool insertPoint(int64_t point_idx) {
        const Point& p = points_[point_idx];

        // Find tetrahedra whose circumsphere contains the point
        std::vector<int64_t> containing_tets;
        for (size_t t = 0; t < tetrahedra_.size(); ++t) {
            if (tetrahedra_[t].valid && inCircumsphere(t, p)) {
                containing_tets.push_back(static_cast<int64_t>(t));
            }
        }

        if (containing_tets.empty()) {
            // Point outside all circumspheres - find containing tetrahedron
            for (size_t t = 0; t < tetrahedra_.size(); ++t) {
                if (tetrahedra_[t].valid && pointInTetrahedron(t, p)) {
                    containing_tets.push_back(static_cast<int64_t>(t));
                    break;
                }
            }
        }

        if (containing_tets.empty()) {
            return false;
        }

        // Mark containing tetrahedra as invalid
        for (int64_t t : containing_tets) {
            tetrahedra_[t].valid = false;
        }

        // Find boundary faces of the hole
        std::vector<std::array<int64_t, 3>> boundary_faces;
        for (int64_t t : containing_tets) {
            for (int f = 0; f < 4; ++f) {
                // Get face vertices
                std::array<int64_t, 3> face;
                int fi = 0;
                for (int i = 0; i < 4; ++i) {
                    if (i != f) {
                        face[fi++] = tetrahedra_[t].v[i];
                    }
                }

                // Check if this face is shared with another containing tetrahedron
                bool shared = false;
                for (int64_t t2 : containing_tets) {
                    if (t2 != t && sharesFace(t2, face)) {
                        shared = true;
                        break;
                    }
                }

                if (!shared) {
                    boundary_faces.push_back(face);
                }
            }
        }

        // Create new tetrahedra from boundary faces to new point
        for (const auto& face : boundary_faces) {
            Tetrahedron new_tet;
            new_tet.v[0] = face[0];
            new_tet.v[1] = face[1];
            new_tet.v[2] = face[2];
            new_tet.v[3] = point_idx;

            // Ensure positive orientation
            if (orient3d(new_tet.v[0], new_tet.v[1], new_tet.v[2], new_tet.v[3]) < 0) {
                std::swap(new_tet.v[0], new_tet.v[1]);
            }

            tetrahedra_.push_back(new_tet);
        }

        return true;
    }

    bool sharesFace(int64_t tet_idx, const std::array<int64_t, 3>& face) const {
        const Tetrahedron& t = tetrahedra_[tet_idx];
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (t.v[i] == face[j]) {
                    count++;
                    break;
                }
            }
        }
        return count == 3;
    }

    bool inCircumsphere(size_t tet_idx, const Point& p) const {
        const Tetrahedron& t = tetrahedra_[tet_idx];
        const Point& a = points_[t.v[0]];
        const Point& b = points_[t.v[1]];
        const Point& c = points_[t.v[2]];
        const Point& d = points_[t.v[3]];

        // Compute determinant for insphere test
        float ax = a.x - p.x, ay = a.y - p.y, az = a.z - p.z;
        float bx = b.x - p.x, by = b.y - p.y, bz = b.z - p.z;
        float cx = c.x - p.x, cy = c.y - p.y, cz = c.z - p.z;
        float dx = d.x - p.x, dy = d.y - p.y, dz = d.z - p.z;

        float ab = ax * ax + ay * ay + az * az;
        float bb = bx * bx + by * by + bz * bz;
        float cb = cx * cx + cy * cy + cz * cz;
        float db = dx * dx + dy * dy + dz * dz;

        float det =
            ax * (by * (cz * db - cb * dz) - bz * (cy * db - cb * dy) + bb * (cy * dz - cz * dy)) -
            ay * (bx * (cz * db - cb * dz) - bz * (cx * db - cb * dx) + bb * (cx * dz - cz * dx)) +
            az * (bx * (cy * db - cb * dy) - by * (cx * db - cb * dx) + bb * (cx * dy - cy * dx)) -
            ab * (bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx));

        // Account for orientation
        float orient = orient3d(t.v[0], t.v[1], t.v[2], t.v[3]);
        if (orient < 0) {
            det = -det;
        }

        return det > 0;
    }

    bool pointInTetrahedron(size_t tet_idx, const Point& p) const {
        const Tetrahedron& t = tetrahedra_[tet_idx];

        // Point is inside if it's on the positive side of all faces
        // (assuming consistent orientation)
        int64_t v0 = t.v[0], v1 = t.v[1], v2 = t.v[2], v3 = t.v[3];

        float d0 = orient3d_point(v1, v2, v3, p);
        float d1 = orient3d_point(v0, v3, v2, p);
        float d2 = orient3d_point(v0, v1, v3, p);
        float d3 = orient3d_point(v0, v2, v1, p);

        bool has_neg = (d0 < 0) || (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool has_pos = (d0 > 0) || (d1 > 0) || (d2 > 0) || (d3 > 0);

        return !(has_neg && has_pos);
    }

    float orient3d(int64_t a, int64_t b, int64_t c, int64_t d) const {
        const Point& pa = points_[a];
        const Point& pb = points_[b];
        const Point& pc = points_[c];
        const Point& pd = points_[d];

        float adx = pa.x - pd.x, ady = pa.y - pd.y, adz = pa.z - pd.z;
        float bdx = pb.x - pd.x, bdy = pb.y - pd.y, bdz = pb.z - pd.z;
        float cdx = pc.x - pd.x, cdy = pc.y - pd.y, cdz = pc.z - pd.z;

        return adx * (bdy * cdz - bdz * cdy) -
               ady * (bdx * cdz - bdz * cdx) +
               adz * (bdx * cdy - bdy * cdx);
    }

    float orient3d_point(int64_t a, int64_t b, int64_t c, const Point& pd) const {
        const Point& pa = points_[a];
        const Point& pb = points_[b];
        const Point& pc = points_[c];

        float adx = pa.x - pd.x, ady = pa.y - pd.y, adz = pa.z - pd.z;
        float bdx = pb.x - pd.x, bdy = pb.y - pd.y, bdz = pb.z - pd.z;
        float cdx = pc.x - pd.x, cdy = pc.y - pd.y, cdz = pc.z - pd.z;

        return adx * (bdy * cdz - bdz * cdy) -
               ady * (bdx * cdz - bdz * cdx) +
               adz * (bdx * cdy - bdy * cdx);
    }

    void removeSuperTetrahedron() {
        // Mark tetrahedra containing super vertices as invalid
        for (auto& t : tetrahedra_) {
            if (!t.valid) continue;

            for (int i = 0; i < 4; ++i) {
                if (t.v[i] >= super_start_idx_) {
                    t.valid = false;
                    break;
                }
            }
        }

        // Remove super vertices from points
        points_.resize(super_start_idx_);
    }

    TriangulationResult buildResult() {
        // Count valid tetrahedra
        size_t num_valid = 0;
        for (const auto& t : tetrahedra_) {
            if (t.valid) ++num_valid;
        }

        // Build output tensors
        core::Tensor tets = core::Tensor::empty(
            {num_valid, 4},
            core::Device::CPU, core::DataType::Int64);

        core::Tensor neighbors = core::Tensor::full(
            {num_valid, 4}, static_cast<float>(-1),
            core::Device::CPU, core::DataType::Int64);

        int64_t* tet_data = tets.ptr<int64_t>();
        size_t out_idx = 0;
        for (const auto& t : tetrahedra_) {
            if (!t.valid) continue;

            for (int i = 0; i < 4; ++i) {
                tet_data[out_idx * 4 + i] = t.v[i];
            }
            ++out_idx;
        }

        return TriangulationResult{
            std::move(tets),
            std::move(neighbors),
            num_valid
        };
    }
};

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

std::expected<TriangulationResult, std::string> triangulate_cpu(
    const core::Tensor& vertices) {

    if (!vertices.is_valid()) {
        return std::unexpected("Invalid vertices tensor");
    }

    if (vertices.ndim() != 2 || vertices.shape()[1] != 3) {
        return std::unexpected("Vertices must be [N, 3] tensor");
    }

    // Copy vertices to CPU if needed
    core::Tensor verts_cpu = vertices.to(core::Device::CPU);
    const float* data = verts_cpu.ptr<float>();
    const size_t num_points = static_cast<size_t>(verts_cpu.shape()[0]);

    // Build point list
    std::vector<IncrementalDelaunay::Point> points;
    points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        points.emplace_back(
            data[i * 3 + 0],
            data[i * 3 + 1],
            data[i * 3 + 2],
            static_cast<int64_t>(i));
    }

    // Run triangulation
    IncrementalDelaunay delaunay(std::move(points));
    return delaunay.triangulate();
}

std::expected<TriangulationResult, std::string> triangulate_gpu(
    const core::Tensor& vertices,
    void* stream) {
#ifdef LFS_HAS_GPU_DELAUNAY
    if (!vertices.is_valid()) {
        return std::unexpected("Invalid vertices tensor");
    }

    if (vertices.ndim() != 2 || vertices.shape()[1] != 3) {
        return std::unexpected("Vertices must be [N, 3] tensor");
    }

    // Copy vertices to CPU for gDel3D input
    core::Tensor verts_cpu = vertices.to(core::Device::CPU);
    const float* data = verts_cpu.ptr<float>();
    const size_t num_points = static_cast<size_t>(verts_cpu.shape()[0]);

    if (num_points < 4) {
        return std::unexpected("Need at least 4 points for tetrahedralization");
    }

    // ------------------------------------------------------------
    // Pre-process points for gDel3D robustness
    // gDel3D fails with "Input too degenerate" when points are:
    // 1. Too close together (duplicates/near-duplicates)
    // 2. Not well-distributed in space
    // 3. Have very different scales
    // ------------------------------------------------------------

    // Step 1: Compute bounding box for normalization
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_points; ++i) {
        float x = data[i * 3 + 0];
        float y = data[i * 3 + 1];
        float z = data[i * 3 + 2];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
    }

    float extent_x = max_x - min_x;
    float extent_y = max_y - min_y;
    float extent_z = max_z - min_z;
    float max_extent = std::max({extent_x, extent_y, extent_z, 1e-6f});

    // Step 2: Filter near-duplicate points using spatial hashing
    // Use a grid cell size that ensures well-separated points
    // Smaller cell = keep more points = match Python radiance_meshes behavior
    // Python uses 142k points for kitchen scene, so we use a finer grid
    constexpr float CELL_SIZE = 1e-5f;  // Grid cell size in normalized space (0.001% of extent)
    std::unordered_set<int64_t> occupied_cells;
    std::vector<size_t> valid_indices;
    valid_indices.reserve(num_points);

    auto hash_cell = [](int cx, int cy, int cz) -> int64_t {
        // Combine cell coordinates into a single hash
        int64_t h = static_cast<int64_t>(cx) * 73856093LL;
        h ^= static_cast<int64_t>(cy) * 19349663LL;
        h ^= static_cast<int64_t>(cz) * 83492791LL;
        return h;
    };

    for (size_t i = 0; i < num_points; ++i) {
        // Normalize to [0, 1] range
        float nx = (data[i * 3 + 0] - min_x) / max_extent;
        float ny = (data[i * 3 + 1] - min_y) / max_extent;
        float nz = (data[i * 3 + 2] - min_z) / max_extent;

        // Compute cell coordinates
        int cx = static_cast<int>(nx / CELL_SIZE);
        int cy = static_cast<int>(ny / CELL_SIZE);
        int cz = static_cast<int>(nz / CELL_SIZE);

        int64_t cell_hash = hash_cell(cx, cy, cz);

        // Only keep point if cell is not occupied
        if (occupied_cells.find(cell_hash) == occupied_cells.end()) {
            occupied_cells.insert(cell_hash);
            valid_indices.push_back(i);
        }
    }

    const size_t filtered_points = valid_indices.size();
    if (filtered_points < 4) {
        return std::unexpected("After duplicate filtering, fewer than 4 points remain");
    }

    LOG_INFO("GPU Delaunay: filtered {} -> {} points (removed {} near-duplicates)",
             num_points, filtered_points, num_points - filtered_points);

    // Step 3: Build Point3HVec with normalized coordinates and jitter
    // gDel3D works best with coordinates in reasonable range [0, 1000]
    constexpr float SCALE_FACTOR = 1000.0f;
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    // Use fixed large jitter to ensure gDel3D's initial 4 extreme points are non-coplanar
    // gDel3D selects leftmost, rightmost, furthest 2D, furthest 3D and checks coplanarity
    // Real-world SfM point clouds often have near-coplanar structures that cause failure
    // A jitter of 1% of scale ensures that even axis-aligned point clouds work
    float jitter_magnitude = SCALE_FACTOR * 0.01f;  // 1% of total scale = 10 units
    std::uniform_real_distribution<float> jitter(-jitter_magnitude, jitter_magnitude);

    LOG_INFO("GPU Delaunay: using jitter magnitude {:.4f} (1% of scale)",
             jitter_magnitude);

    Point3HVec point_vec;
    // Reserve space for filtered points + 4 anchor points
    point_vec.reserve(filtered_points + 4);

    // gDel3D selects extreme points (leftmost, rightmost, furthest 2D/3D) and checks
    // if they're coplanar. Add 4 anchor points at the corners of a tetrahedron
    // that spans FAR outside the main point cloud, ensuring they will be the extreme
    // points and are guaranteed non-coplanar.
    constexpr float MARGIN = 10000.0f;  // VERY large margin (10x scale) to ensure anchors are extreme
    Point3 anchor0, anchor1, anchor2, anchor3;
    // Tetrahedron vertices well outside the main point cloud
    // anchor0: leftmost (smallest X)
    anchor0._p[0] = -MARGIN;               anchor0._p[1] = SCALE_FACTOR * 0.5f;   anchor0._p[2] = SCALE_FACTOR * 0.5f;
    // anchor1: rightmost (largest X)
    anchor1._p[0] = SCALE_FACTOR + MARGIN; anchor1._p[1] = SCALE_FACTOR * 0.5f;   anchor1._p[2] = SCALE_FACTOR * 0.5f;
    // anchor2: furthest in Y (for 2D distance from X-axis)
    anchor2._p[0] = SCALE_FACTOR * 0.5f;   anchor2._p[1] = SCALE_FACTOR + MARGIN; anchor2._p[2] = SCALE_FACTOR * 0.5f;
    // anchor3: furthest in Z (for 3D distance from XY plane)
    anchor3._p[0] = SCALE_FACTOR * 0.5f;   anchor3._p[1] = SCALE_FACTOR * 0.5f;   anchor3._p[2] = SCALE_FACTOR + MARGIN;
    point_vec.push_back(anchor0);
    point_vec.push_back(anchor1);
    point_vec.push_back(anchor2);
    point_vec.push_back(anchor3);

    // Map from point_vec index to original index
    // First 4 are anchor points (map to dummy index num_points)
    std::vector<size_t> index_map;
    index_map.reserve(filtered_points + 4);
    index_map.push_back(num_points);  // anchor0 -> dummy
    index_map.push_back(num_points);  // anchor1 -> dummy
    index_map.push_back(num_points);  // anchor2 -> dummy
    index_map.push_back(num_points);  // anchor3 -> dummy

    for (size_t fi = 0; fi < filtered_points; ++fi) {
        size_t i = valid_indices[fi];
        index_map.push_back(i);

        // Normalize to [0, 1] then scale to [0, SCALE_FACTOR]
        float nx = ((data[i * 3 + 0] - min_x) / max_extent) * SCALE_FACTOR;
        float ny = ((data[i * 3 + 1] - min_y) / max_extent) * SCALE_FACTOR;
        float nz = ((data[i * 3 + 2] - min_z) / max_extent) * SCALE_FACTOR;

        Point3 p;
        p._p[0] = static_cast<RealType>(nx + jitter(rng));
        p._p[1] = static_cast<RealType>(ny + jitter(rng));
        p._p[2] = static_cast<RealType>(nz + jitter(rng));
        point_vec.push_back(p);
    }

    const size_t total_points = point_vec.size();
    LOG_INFO("GPU Delaunay: {} points total ({} data + 4 anchors)", total_points, filtered_points);

    // Create GPU Delaunay triangulator
    // Enable verbose mode to debug point selection
    GDelParams params(true, false, false, false, InsCentroid);  // verbose=true
    GpuDel triangulator(params);
    GDelOutput output;

    // Compute triangulation
    // gDel3D sets _degenerate=true and returns early if input is degenerate
    LOG_INFO("Running gDel3D GPU Delaunay on {} points...", total_points);
    triangulator.compute(point_vec, &output);

    // Check for failures
    if (!output.failVertVec.empty()) {
        LOG_WARN("GPU Delaunay: {} failed vertices", output.failVertVec.size());
    }

    // Check if output is valid
    const size_t raw_num_tets = output.tetVec.size();
    if (raw_num_tets == 0) {
        return std::unexpected("GPU Delaunay produced 0 tetrahedra");
    }

    // Filter out tetrahedra that reference:
    // 1. Anchor points (indices 0-3) - artificial points we added for robustness
    // 2. Infinity point (index >= total_points) - gDel3D adds an internal "point at infinity"
    // This mirrors pyGDel3D's filtering: indices_np[(indices_np < verts.shape[0]).all(axis=1)]
    std::vector<std::array<int64_t, 4>> valid_tets;
    valid_tets.reserve(raw_num_tets);

    const int max_valid_idx = static_cast<int>(total_points);
    for (size_t t = 0; t < raw_num_tets; ++t) {
        const Tet& tet = output.tetVec[t];
        // Check bounds: skip if any vertex is an anchor (< 4) or infinity point (>= total_points)
        if (tet._v[0] < 4 || tet._v[1] < 4 || tet._v[2] < 4 || tet._v[3] < 4) {
            continue;
        }
        if (tet._v[0] >= max_valid_idx || tet._v[1] >= max_valid_idx ||
            tet._v[2] >= max_valid_idx || tet._v[3] >= max_valid_idx) {
            continue;
        }
        // Map back to original indices using index_map
        valid_tets.push_back({
            static_cast<int64_t>(index_map[tet._v[0]]),
            static_cast<int64_t>(index_map[tet._v[1]]),
            static_cast<int64_t>(index_map[tet._v[2]]),
            static_cast<int64_t>(index_map[tet._v[3]])
        });
    }

    const size_t num_tets = valid_tets.size();
    if (num_tets == 0) {
        return std::unexpected("GPU Delaunay produced 0 valid tetrahedra after filtering anchor points");
    }

    LOG_INFO("GPU Delaunay: {} raw tets, {} valid tets after filtering anchors",
             raw_num_tets, num_tets);

    // Build output tensors
    core::Tensor tets = core::Tensor::empty(
        {num_tets, 4}, core::Device::CPU, core::DataType::Int64);

    int64_t* tet_data = tets.ptr<int64_t>();
    for (size_t t = 0; t < num_tets; ++t) {
        tet_data[t * 4 + 0] = valid_tets[t][0];
        tet_data[t * 4 + 1] = valid_tets[t][1];
        tet_data[t * 4 + 2] = valid_tets[t][2];
        tet_data[t * 4 + 3] = valid_tets[t][3];
    }

    // Move to GPU
    tets = tets.to(core::Device::CUDA);

    // Neighbor information is invalidated by filtering, so create empty tensor
    core::Tensor neighbors = core::Tensor::full(
        {num_tets, 4}, static_cast<float>(-1),
        core::Device::CUDA, core::DataType::Int64);

    LOG_INFO("GPU Delaunay: {} points -> {} tetrahedra in {:.2f}ms",
             filtered_points, num_tets, output.stats.totalTime);

    return TriangulationResult{
        std::move(tets),
        std::move(neighbors),
        num_tets
    };

#else
    return std::unexpected("GPU Delaunay not available - gDel3D not linked. Build with LFS_HAS_GPU_DELAUNAY=ON");
#endif
}

core::Tensor create_exterior_shell(
    const core::Tensor& vertices,
    float expansion_factor,
    int num_shell_points) {

    if (!vertices.is_valid() || vertices.shape()[0] == 0) {
        return core::Tensor();
    }

    core::Tensor verts_cpu = vertices.to(core::Device::CPU);
    const float* data = verts_cpu.ptr<float>();
    const size_t num_points = static_cast<size_t>(verts_cpu.shape()[0]);

    // Compute bounding box
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_points; ++i) {
        float x = data[i * 3 + 0];
        float y = data[i * 3 + 1];
        float z = data[i * 3 + 2];
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
    }

    // Compute center and radius
    float cx = (min_x + max_x) * 0.5f;
    float cy = (min_y + max_y) * 0.5f;
    float cz = (min_z + max_z) * 0.5f;

    float dx = (max_x - min_x) * 0.5f * expansion_factor;
    float dy = (max_y - min_y) * 0.5f * expansion_factor;
    float dz = (max_z - min_z) * 0.5f * expansion_factor;

    // Create shell vertices (8 corners of expanded bounding box)
    core::Tensor shell = core::Tensor::empty(
        {static_cast<size_t>(num_shell_points), 3}, core::Device::CPU, core::DataType::Float32);

    float* shell_data = shell.ptr<float>();

    if (num_shell_points == 8) {
        // Cube corners
        const float signs[8][3] = {
            {-1, -1, -1}, {-1, -1, +1}, {-1, +1, -1}, {-1, +1, +1},
            {+1, -1, -1}, {+1, -1, +1}, {+1, +1, -1}, {+1, +1, +1}
        };
        for (int i = 0; i < 8; ++i) {
            shell_data[i * 3 + 0] = cx + signs[i][0] * dx;
            shell_data[i * 3 + 1] = cy + signs[i][1] * dy;
            shell_data[i * 3 + 2] = cz + signs[i][2] * dz;
        }
    } else {
        // Distribute points on sphere
        float radius = std::max({dx, dy, dz});
        for (int i = 0; i < num_shell_points; ++i) {
            float phi = static_cast<float>(i) / num_shell_points * 2.0f * 3.14159265f;
            float theta = std::acos(1.0f - 2.0f * (i + 0.5f) / num_shell_points);
            shell_data[i * 3 + 0] = cx + radius * std::sin(theta) * std::cos(phi);
            shell_data[i * 3 + 1] = cy + radius * std::sin(theta) * std::sin(phi);
            shell_data[i * 3 + 2] = cz + radius * std::cos(theta);
        }
    }

    return shell;
}

std::expected<void, std::string> validate_mesh(
    const core::Tensor& vertices,
    const core::Tensor& tetrahedra) {

    if (!vertices.is_valid() || !tetrahedra.is_valid()) {
        return std::unexpected("Invalid tensors");
    }

    core::Tensor verts_cpu = vertices.to(core::Device::CPU);
    core::Tensor tets_cpu = tetrahedra.to(core::Device::CPU);

    const float* vert_data = verts_cpu.ptr<float>();
    const int64_t* tet_data = tets_cpu.ptr<int64_t>();

    const size_t num_verts = static_cast<size_t>(verts_cpu.shape()[0]);
    const size_t num_tets = static_cast<size_t>(tets_cpu.shape()[0]);

    size_t degenerate_count = 0;
    size_t inverted_count = 0;

    for (size_t t = 0; t < num_tets; ++t) {
        int64_t v0 = tet_data[t * 4 + 0];
        int64_t v1 = tet_data[t * 4 + 1];
        int64_t v2 = tet_data[t * 4 + 2];
        int64_t v3 = tet_data[t * 4 + 3];

        // Check bounds
        if (v0 < 0 || v0 >= static_cast<int64_t>(num_verts) ||
            v1 < 0 || v1 >= static_cast<int64_t>(num_verts) ||
            v2 < 0 || v2 >= static_cast<int64_t>(num_verts) ||
            v3 < 0 || v3 >= static_cast<int64_t>(num_verts)) {
            return std::unexpected("Tetrahedron " + std::to_string(t) + " has invalid vertex index");
        }

        // Compute signed volume
        const float* p0 = vert_data + v0 * 3;
        const float* p1 = vert_data + v1 * 3;
        const float* p2 = vert_data + v2 * 3;
        const float* p3 = vert_data + v3 * 3;

        float ax = p1[0] - p0[0], ay = p1[1] - p0[1], az = p1[2] - p0[2];
        float bx = p2[0] - p0[0], by = p2[1] - p0[1], bz = p2[2] - p0[2];
        float cx = p3[0] - p0[0], cy = p3[1] - p0[1], cz = p3[2] - p0[2];

        float vol = ax * (by * cz - bz * cy) -
                    ay * (bx * cz - bz * cx) +
                    az * (bx * cy - by * cx);

        if (std::abs(vol) < 1e-10f) {
            degenerate_count++;
        } else if (vol < 0) {
            inverted_count++;
        }
    }

    if (degenerate_count > 0 || inverted_count > 0) {
        return std::unexpected(
            "Mesh has " + std::to_string(degenerate_count) + " degenerate and " +
            std::to_string(inverted_count) + " inverted tetrahedra");
    }

    return {};
}

} // namespace lfs::tetra::delaunay
