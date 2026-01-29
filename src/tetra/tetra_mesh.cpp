/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_mesh.hpp"
#include "tetra/tetra_delaunay.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_map>

namespace lfs::tetra {

// ------------------------------
// LIFECYCLE
// ------------------------------

TetraMesh::~TetraMesh() = default;

TetraMesh::TetraMesh(TetraMesh&& other) noexcept
    : scene_scale_(other.scene_scale_)
    , vertices_(std::move(other.vertices_))
    , ext_vertices_(std::move(other.ext_vertices_))
    , tetrahedra_(std::move(other.tetrahedra_))
    , visibility_scores_(std::move(other.visibility_scores_))
    , density_(std::move(other.density_))
    , base_color_(std::move(other.base_color_))
    , gradient_(std::move(other.gradient_)) {
    other.scene_scale_ = 0.f;
}

TetraMesh& TetraMesh::operator=(TetraMesh&& other) noexcept {
    if (this != &other) {
        scene_scale_ = other.scene_scale_;
        vertices_ = std::move(other.vertices_);
        ext_vertices_ = std::move(other.ext_vertices_);
        tetrahedra_ = std::move(other.tetrahedra_);
        visibility_scores_ = std::move(other.visibility_scores_);
        density_ = std::move(other.density_);
        base_color_ = std::move(other.base_color_);
        gradient_ = std::move(other.gradient_);
        other.scene_scale_ = 0.f;
    }
    return *this;
}

TetraMesh::TetraMesh(core::Tensor vertices,
                     core::Tensor ext_vertices,
                     core::Tensor tetrahedra,
                     float scene_scale)
    : scene_scale_(scene_scale)
    , vertices_(std::move(vertices))
    , ext_vertices_(std::move(ext_vertices))
    , tetrahedra_(std::move(tetrahedra)) {}

// ------------------------------
// FACTORY METHODS
// ------------------------------

std::expected<TetraMesh, std::string> TetraMesh::from_point_cloud(
    const core::PointCloud& point_cloud,
    float scene_scale,
    float shell_expansion) {

    if (!point_cloud.means.is_valid() || point_cloud.means.shape()[0] == 0) {
        return std::unexpected("Point cloud is empty");
    }

    const size_t original_points = static_cast<size_t>(point_cloud.means.shape()[0]);
    LOG_INFO("Creating TetraMesh from point cloud with {} points", original_points);

    // Downsample if too many points (Delaunay is O(n log n) but memory intensive)
    // Radiance Meshes paper uses ~100k-300k points typically
    // Python reference uses 142k for kitchen scene, so we allow up to 300k
    constexpr size_t MAX_POINTS = 300000;
    core::Tensor vertices;

    if (original_points > MAX_POINTS) {
        LOG_INFO("Downsampling point cloud from {} to {} points", original_points, MAX_POINTS);

        // Random downsampling - copy to CPU for index selection
        core::Tensor means_cpu = point_cloud.means.to(core::Device::CPU);
        const float* src_data = means_cpu.ptr<float>();

        // Generate random indices
        std::vector<size_t> indices(original_points);
        for (size_t i = 0; i < original_points; ++i) {
            indices[i] = i;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
        indices.resize(MAX_POINTS);

        // Create downsampled tensor
        core::Tensor downsampled = core::Tensor::empty(
            {MAX_POINTS, 3}, core::Device::CPU, core::DataType::Float32);
        float* dst_data = downsampled.ptr<float>();

        for (size_t i = 0; i < MAX_POINTS; ++i) {
            size_t src_idx = indices[i];
            dst_data[i * 3 + 0] = src_data[src_idx * 3 + 0];
            dst_data[i * 3 + 1] = src_data[src_idx * 3 + 1];
            dst_data[i * 3 + 2] = src_data[src_idx * 3 + 2];
        }

        vertices = downsampled.to(core::Device::CUDA);
    } else {
        vertices = point_cloud.means.clone();
    }

    // Create exterior shell (returns CPU tensor)
    core::Tensor ext_vertices = delaunay::create_exterior_shell(vertices, shell_expansion);

    LOG_INFO("Created {} exterior shell vertices", ext_vertices.shape()[0]);

    // Concatenate all vertices for triangulation
    // Both need to be on CPU for gDel3D input preparation
    core::Tensor vertices_cpu = vertices.to(core::Device::CPU);
    core::Tensor ext_vertices_cpu = ext_vertices.to(core::Device::CPU);
    core::Tensor all_vertices = core::Tensor::cat({vertices_cpu, ext_vertices_cpu}, 0);

    // Compute Delaunay triangulation on GPU
    // GPU Delaunay (gDel3D) with pre-processing for robustness
    LOG_INFO("Starting GPU Delaunay triangulation with {} vertices...", all_vertices.shape()[0]);
    auto tri_result = delaunay::triangulate_gpu(all_vertices);
    if (!tri_result) {
        return std::unexpected("GPU Delaunay triangulation failed: " + tri_result.error());
    }

    LOG_INFO("Created {} tetrahedra", tri_result->num_tetrahedra);

    // Validate mesh quality
    auto validation = delaunay::validate_mesh(all_vertices, tri_result->tetrahedra);
    if (!validation) {
        LOG_WARN("Mesh validation warning: {}", validation.error());
    }

    // Ensure vertices and ext_vertices are on the same device (CUDA)
    core::Tensor vertices_cuda = vertices.to(core::Device::CUDA);
    core::Tensor ext_vertices_cuda = ext_vertices.to(core::Device::CUDA);

    TetraMesh mesh(std::move(vertices_cuda),
                   std::move(ext_vertices_cuda),
                   std::move(tri_result->tetrahedra),
                   scene_scale);

    // Initialize per-tetrahedron learnable parameters
    mesh.init_per_tet_params();

    return mesh;
}

// ------------------------------
// COMPUTED GETTERS
// ------------------------------

core::Tensor TetraMesh::get_all_vertices() const {
    if (!vertices_.is_valid() && !ext_vertices_.is_valid()) {
        return core::Tensor();
    }
    if (!vertices_.is_valid()) {
        return ext_vertices_;
    }
    if (!ext_vertices_.is_valid()) {
        return vertices_;
    }
    return core::Tensor::cat({vertices_, ext_vertices_}, 0);
}

// ------------------------------
// VERTEX MODIFICATION
// ------------------------------

std::expected<void, std::string> TetraMesh::add_vertices(const core::Tensor& new_vertices) {
    if (!new_vertices.is_valid() || new_vertices.shape()[0] == 0) {
        return {};  // Nothing to add
    }

    if (new_vertices.ndim() != 2 || new_vertices.shape()[1] != 3) {
        return std::unexpected("new_vertices must have shape [N, 3]");
    }

    // Ensure new vertices are on CUDA
    core::Tensor new_verts_cuda = new_vertices.to(core::Device::CUDA);

    if (!vertices_.is_valid()) {
        // First vertices
        vertices_ = std::move(new_verts_cuda);
    } else {
        // Concatenate with existing vertices
        vertices_ = core::Tensor::cat({vertices_, new_verts_cuda}, 0);
    }

    LOG_INFO("Added {} vertices, total interior vertices: {}",
             new_vertices.shape()[0], vertices_.shape()[0]);

    return {};
}

// ------------------------------
// TRIANGULATION UPDATE
// ------------------------------

std::expected<void, std::string> TetraMesh::update_triangulation() {
    if (!vertices_.is_valid()) {
        return std::unexpected("No vertices to triangulate");
    }

    // Concatenate interior and exterior vertices
    core::Tensor all_vertices = get_all_vertices();

    // Recompute Delaunay triangulation on GPU (no CPU fallback)
    auto tri_result = delaunay::triangulate_gpu(all_vertices);
    if (!tri_result) {
        return std::unexpected("GPU re-triangulation failed: " + tri_result.error());
    }

    size_t old_num_tets = num_tetrahedra();
    tetrahedra_ = std::move(tri_result->tetrahedra);
    size_t new_num_tets = num_tetrahedra();

    // Only reinitialize per-tet params if this is the first time or if they don't exist
    // Don't reset learned parameters during normal training updates
    if (!density_.is_valid() || density_.shape()[0] != new_num_tets) {
        LOG_INFO("Resizing per-tet params: {} -> {} tetrahedra", old_num_tets, new_num_tets);
        init_per_tet_params();
    }

    return {};
}

void TetraMesh::init_per_tet_params() {
    size_t T = num_tetrahedra();
    if (T == 0) {
        return;
    }

    // Initialize density to small positive value (log scale)
    density_ = core::Tensor::full({T}, -2.0f, core::Device::CUDA, core::DataType::Float32);

    // Initialize base_color to gray
    base_color_ = core::Tensor::full({T, 3}, 0.5f, core::Device::CUDA, core::DataType::Float32);

    // Initialize gradient to zero
    gradient_ = core::Tensor::zeros({T, 3}, core::Device::CUDA, core::DataType::Float32);

    LOG_INFO("Initialized per-tet parameters for {} tetrahedra", T);
}

// ------------------------------
// MESH EXTRACTION
// ------------------------------

std::expected<std::tuple<core::Tensor, core::Tensor, core::Tensor>, std::string>
TetraMesh::extract_surface_mesh(float visibility_threshold) const {
    if (!tetrahedra_.is_valid() || num_tetrahedra() == 0) {
        return std::unexpected("No tetrahedra to extract surface from");
    }

    core::Tensor all_verts = get_all_vertices();
    const size_t num_tets = num_tetrahedra();

    // Count boundary triangles
    // A triangle is on the boundary if it belongs to only one tetrahedron
    // For now, extract all faces and deduplicate

    // Each tetrahedron has 4 triangular faces
    // Face ordering: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    constexpr int face_indices[4][3] = {
        {0, 2, 1},  // Outward-facing normals
        {0, 1, 3},
        {0, 3, 2},
        {1, 2, 3}
    };

    // Get tetrahedra on CPU for face extraction
    core::Tensor tets_cpu = tetrahedra_.to(core::Device::CPU);
    const int64_t* tet_data = tets_cpu.ptr<int64_t>();

    // Build face adjacency map
    // Key: sorted vertex indices, Value: list of (tet_idx, face_idx)
    std::unordered_map<uint64_t, std::vector<std::pair<size_t, int>>> face_map;

    auto hash_face = [](int64_t v0, int64_t v1, int64_t v2) -> uint64_t {
        // Sort indices
        if (v0 > v1) std::swap(v0, v1);
        if (v1 > v2) std::swap(v1, v2);
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<uint64_t>(v0) << 40) |
               (static_cast<uint64_t>(v1) << 20) |
               static_cast<uint64_t>(v2);
    };

    for (size_t t = 0; t < num_tets; ++t) {
        const int64_t* tet = tet_data + t * 4;
        for (int f = 0; f < 4; ++f) {
            int64_t v0 = tet[face_indices[f][0]];
            int64_t v1 = tet[face_indices[f][1]];
            int64_t v2 = tet[face_indices[f][2]];
            uint64_t key = hash_face(v0, v1, v2);
            face_map[key].emplace_back(t, f);
        }
    }

    // Extract boundary faces (faces shared by only one tetrahedron)
    std::vector<std::array<int64_t, 3>> boundary_faces;
    for (const auto& [key, faces] : face_map) {
        if (faces.size() == 1) {
            // This is a boundary face
            size_t t = faces[0].first;
            int f = faces[0].second;
            const int64_t* tet = tet_data + t * 4;
            boundary_faces.push_back({
                tet[face_indices[f][0]],
                tet[face_indices[f][1]],
                tet[face_indices[f][2]]
            });
        }
    }

    if (boundary_faces.empty()) {
        return std::unexpected("No boundary faces found");
    }

    // Create output tensors
    core::Tensor triangles = core::Tensor::empty(
        {boundary_faces.size(), 3},
        core::Device::CPU, core::DataType::Int64);

    int64_t* tri_data = triangles.ptr<int64_t>();
    for (size_t i = 0; i < boundary_faces.size(); ++i) {
        tri_data[i * 3 + 0] = boundary_faces[i][0];
        tri_data[i * 3 + 1] = boundary_faces[i][1];
        tri_data[i * 3 + 2] = boundary_faces[i][2];
    }

    // Use vertex positions and default white color
    core::Tensor colors = core::Tensor::full(
        {all_verts.shape()[0], 3}, 1.0f,
        core::Device::CPU, core::DataType::Float32);

    return std::make_tuple(
        all_verts.to(core::Device::CPU),
        std::move(triangles),
        std::move(colors));
}

// ------------------------------
// SERIALIZATION
// ------------------------------

void TetraMesh::serialize(std::ostream& os) const {
    // Write magic number and version
    const uint32_t magic = 0x54455452;  // "TETR"
    const uint32_t version = 1;
    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write scene scale
    os.write(reinterpret_cast<const char*>(&scene_scale_), sizeof(scene_scale_));

    // Serialize tensors using stream operator
    auto write_tensor = [&os](const core::Tensor& t) {
        bool valid = t.is_valid();
        os.write(reinterpret_cast<const char*>(&valid), sizeof(valid));
        if (valid) {
            os << t;
        }
    };

    write_tensor(vertices_);
    write_tensor(ext_vertices_);
    write_tensor(tetrahedra_);
}

void TetraMesh::deserialize(std::istream& is) {
    // Read and verify magic number
    uint32_t magic;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x54455452) {
        throw std::runtime_error("Invalid TetraMesh file format");
    }

    // Read version
    uint32_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported TetraMesh version: " + std::to_string(version));
    }

    // Read scene scale
    is.read(reinterpret_cast<char*>(&scene_scale_), sizeof(scene_scale_));

    // Deserialize tensors using stream operator
    auto read_tensor = [&is]() -> core::Tensor {
        bool valid;
        is.read(reinterpret_cast<char*>(&valid), sizeof(valid));
        if (valid) {
            core::Tensor t;
            is >> t;
            return t;
        }
        return core::Tensor();
    };

    vertices_ = read_tensor();
    ext_vertices_ = read_tensor();
    tetrahedra_ = read_tensor();
}

void TetraMesh::save_ply(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }

    core::Tensor all_verts = get_all_vertices();
    core::Tensor verts_cpu = all_verts.to(core::Device::CPU);
    core::Tensor tets_cpu = tetrahedra_.to(core::Device::CPU);

    const size_t num_verts = static_cast<size_t>(verts_cpu.shape()[0]);
    const size_t num_tets = static_cast<size_t>(tets_cpu.shape()[0]);

    // Write PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << num_verts << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element tetrahedron " << num_tets << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // Write vertices
    const float* vert_data = verts_cpu.ptr<float>();
    file.write(reinterpret_cast<const char*>(vert_data), num_verts * 3 * sizeof(float));

    // Write tetrahedra
    const int64_t* tet_data = tets_cpu.ptr<int64_t>();
    for (size_t t = 0; t < num_tets; ++t) {
        uint8_t count = 4;
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (int i = 0; i < 4; ++i) {
            int32_t idx = static_cast<int32_t>(tet_data[t * 4 + i]);
            file.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        }
    }
}

void TetraMesh::save_ply(const std::filesystem::path& path,
                         const TetraPlyFeatures& features) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }

    // ------------------------------------------------------------
    // Prepare data on CPU
    // ------------------------------------------------------------
    core::Tensor all_verts = get_all_vertices();
    core::Tensor verts_cpu = all_verts.to(core::Device::CPU);
    core::Tensor tets_cpu = tetrahedra_.to(core::Device::CPU);

    const size_t num_verts = static_cast<size_t>(verts_cpu.shape()[0]);
    const size_t num_tets = static_cast<size_t>(tets_cpu.shape()[0]);

    // Prepare feature tensors on CPU
    core::Tensor density_cpu = features.density.to(core::Device::CPU);
    core::Tensor gradient_cpu = features.gradient.to(core::Device::CPU);
    core::Tensor sh_dc_cpu = features.sh_dc.to(core::Device::CPU);
    core::Tensor sh_rest_cpu = features.sh_rest.to(core::Device::CPU);

    // Determine SH dimensions (typically 15 for degree 3)
    const size_t sh_dim = sh_rest_cpu.is_valid() && sh_rest_cpu.ndim() >= 2
        ? static_cast<size_t>(sh_rest_cpu.shape()[1])
        : 0;

    // ------------------------------------------------------------
    // Write PLY header (radiance_meshes format)
    // ------------------------------------------------------------
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";

    // Vertex element
    file << "element vertex " << num_verts << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";

    // Tetrahedron element with features
    file << "element tetrahedron " << num_tets << "\n";
    file << "property list uchar int indices\n";
    file << "property float s\n";  // density
    file << "property float grd_x\n";
    file << "property float grd_y\n";
    file << "property float grd_z\n";
    file << "property float sh_0_r\n";
    file << "property float sh_0_g\n";
    file << "property float sh_0_b\n";

    // Higher order SH coefficients (sh_1 through sh_N)
    for (size_t i = 0; i < sh_dim; ++i) {
        file << "property float sh_" << (i + 1) << "_r\n";
        file << "property float sh_" << (i + 1) << "_g\n";
        file << "property float sh_" << (i + 1) << "_b\n";
    }

    file << "end_header\n";

    // ------------------------------------------------------------
    // Write vertex data (binary)
    // ------------------------------------------------------------
    const float* vert_data = verts_cpu.ptr<float>();
    file.write(reinterpret_cast<const char*>(vert_data),
               static_cast<std::streamsize>(num_verts * 3 * sizeof(float)));

    // ------------------------------------------------------------
    // Write tetrahedron data (binary)
    // ------------------------------------------------------------
    const int64_t* tet_data = tets_cpu.ptr<int64_t>();
    const float* density_data = density_cpu.ptr<float>();
    const float* gradient_data = gradient_cpu.ptr<float>();
    const float* sh_dc_data = sh_dc_cpu.ptr<float>();
    const float* sh_rest_data = sh_rest_cpu.is_valid() ? sh_rest_cpu.ptr<float>() : nullptr;

    for (size_t t = 0; t < num_tets; ++t) {
        // Write indices (list format: count followed by indices)
        uint8_t count = 4;
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (int i = 0; i < 4; ++i) {
            int32_t idx = static_cast<int32_t>(tet_data[t * 4 + i]);
            file.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        }

        // Write density (s)
        float density = density_data[t];
        file.write(reinterpret_cast<const char*>(&density), sizeof(density));

        // Write gradient (grd_x, grd_y, grd_z)
        float grd_x = gradient_data[t * 3 + 0];
        float grd_y = gradient_data[t * 3 + 1];
        float grd_z = gradient_data[t * 3 + 2];
        file.write(reinterpret_cast<const char*>(&grd_x), sizeof(grd_x));
        file.write(reinterpret_cast<const char*>(&grd_y), sizeof(grd_y));
        file.write(reinterpret_cast<const char*>(&grd_z), sizeof(grd_z));

        // Write DC SH (sh_0_r, sh_0_g, sh_0_b)
        float sh_0_r = sh_dc_data[t * 3 + 0];
        float sh_0_g = sh_dc_data[t * 3 + 1];
        float sh_0_b = sh_dc_data[t * 3 + 2];
        file.write(reinterpret_cast<const char*>(&sh_0_r), sizeof(sh_0_r));
        file.write(reinterpret_cast<const char*>(&sh_0_g), sizeof(sh_0_g));
        file.write(reinterpret_cast<const char*>(&sh_0_b), sizeof(sh_0_b));

        // Write higher order SH coefficients (sh_1..sh_N)
        // sh_rest is [T, sh_dim, 3] so index is t * sh_dim * 3 + i * 3 + channel
        if (sh_rest_data != nullptr) {
            for (size_t i = 0; i < sh_dim; ++i) {
                float sh_r = sh_rest_data[t * sh_dim * 3 + i * 3 + 0];
                float sh_g = sh_rest_data[t * sh_dim * 3 + i * 3 + 1];
                float sh_b = sh_rest_data[t * sh_dim * 3 + i * 3 + 2];
                file.write(reinterpret_cast<const char*>(&sh_r), sizeof(sh_r));
                file.write(reinterpret_cast<const char*>(&sh_g), sizeof(sh_g));
                file.write(reinterpret_cast<const char*>(&sh_b), sizeof(sh_b));
            }
        }
    }

    LOG_INFO("Saved PLY with {} vertices, {} tetrahedra, {} SH coefficients per tet",
             num_verts, num_tets, sh_dim + 1);
}

std::expected<TetraMesh, std::string> TetraMesh::load_ply(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected("Failed to open file: " + path.string());
    }

    // Parse PLY header
    std::string line;
    size_t num_vertices = 0;
    size_t num_tetrahedra = 0;
    bool in_header = true;

    while (in_header && std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %zu", &num_vertices);
        } else if (line.find("element tetrahedron") != std::string::npos) {
            sscanf(line.c_str(), "element tetrahedron %zu", &num_tetrahedra);
        } else if (line == "end_header") {
            in_header = false;
        }
    }

    if (num_vertices == 0) {
        return std::unexpected("No vertices found in PLY file");
    }

    // Read vertices
    core::Tensor vertices = core::Tensor::empty(
        {num_vertices, 3},
        core::Device::CPU, core::DataType::Float32);
    file.read(reinterpret_cast<char*>(vertices.ptr<float>()),
              num_vertices * 3 * sizeof(float));

    // Read tetrahedra if present
    core::Tensor tetrahedra;
    if (num_tetrahedra > 0) {
        tetrahedra = core::Tensor::empty(
            {num_tetrahedra, 4},
            core::Device::CPU, core::DataType::Int64);

        int64_t* tet_data = tetrahedra.ptr<int64_t>();
        for (size_t t = 0; t < num_tetrahedra; ++t) {
            uint8_t count;
            file.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (count != 4) {
                return std::unexpected("Expected tetrahedra with 4 vertices");
            }
            for (int i = 0; i < 4; ++i) {
                int32_t idx;
                file.read(reinterpret_cast<char*>(&idx), sizeof(idx));
                tet_data[t * 4 + i] = idx;
            }
        }
    }

    // For now, treat all vertices as interior (no exterior shell)
    return TetraMesh(std::move(vertices),
                     core::Tensor(),
                     std::move(tetrahedra),
                     1.0f);
}

void TetraMesh::export_triangle_mesh(const std::filesystem::path& path,
                                     float visibility_threshold) const {
    auto result = extract_surface_mesh(visibility_threshold);
    if (!result) {
        throw std::runtime_error("Failed to extract surface: " + result.error());
    }

    auto& [vertices, triangles, colors] = *result;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }

    const size_t num_verts = static_cast<size_t>(vertices.shape()[0]);
    const size_t num_tris = static_cast<size_t>(triangles.shape()[0]);

    // Write PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << num_verts << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << num_tris << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // Write vertices with colors
    const float* vert_data = std::get<0>(*result).ptr<float>();
    const float* color_data = std::get<2>(*result).ptr<float>();
    for (size_t v = 0; v < num_verts; ++v) {
        file.write(reinterpret_cast<const char*>(vert_data + v * 3), 3 * sizeof(float));
        uint8_t rgb[3] = {
            static_cast<uint8_t>(std::clamp(color_data[v * 3 + 0] * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(color_data[v * 3 + 1] * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(color_data[v * 3 + 2] * 255.0f, 0.0f, 255.0f))
        };
        file.write(reinterpret_cast<const char*>(rgb), 3);
    }

    // Write faces
    const int64_t* tri_data = std::get<1>(*result).ptr<int64_t>();
    for (size_t t = 0; t < num_tris; ++t) {
        uint8_t count = 3;
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (int i = 0; i < 3; ++i) {
            int32_t idx = static_cast<int32_t>(tri_data[t * 3 + i]);
            file.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        }
    }
}

} // namespace lfs::tetra
