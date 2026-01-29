/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tetra_intersect.cu
 * @brief CUDA kernels for ray-tetrahedron intersection
 *
 * Implements ray-tetrahedron intersection using Plucker coordinates
 * for efficient GPU computation.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace lfs::tetra::cuda {

namespace {

/**
 * @brief Compute Plucker coordinates for a line segment
 *
 * Plucker coordinates represent a line in 3D as a 6-tuple (d, m) where:
 * - d = p1 - p0 (direction)
 * - m = p0 × p1 (moment)
 */
__device__ __forceinline__ void compute_plucker(
    const float* p0, const float* p1, float* plucker) {

    // Direction
    plucker[0] = p1[0] - p0[0];
    plucker[1] = p1[1] - p0[1];
    plucker[2] = p1[2] - p0[2];

    // Moment (cross product)
    plucker[3] = p0[1] * p1[2] - p0[2] * p1[1];
    plucker[4] = p0[2] * p1[0] - p0[0] * p1[2];
    plucker[5] = p0[0] * p1[1] - p0[1] * p1[0];
}

/**
 * @brief Compute side operator for two Plucker lines
 *
 * The side operator determines the relative position of two lines:
 * - side > 0: lines are right-handed
 * - side < 0: lines are left-handed
 * - side = 0: lines intersect or are parallel
 */
__device__ __forceinline__ float plucker_side(
    const float* l1, const float* l2) {

    return l1[0] * l2[3] + l1[1] * l2[4] + l1[2] * l2[5] +
           l1[3] * l2[0] + l1[4] * l2[1] + l1[5] * l2[2];
}

/**
 * @brief Moller-Trumbore ray-triangle intersection
 */
__device__ bool ray_triangle_intersect(
    const float* ray_origin,
    const float* ray_dir,
    const float* v0,
    const float* v1,
    const float* v2,
    float& t,
    float& u,
    float& v) {

    constexpr float EPSILON = 1e-8f;

    // Edge vectors
    float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

    // Cross product ray_dir × e2
    float h[3] = {
        ray_dir[1] * e2[2] - ray_dir[2] * e2[1],
        ray_dir[2] * e2[0] - ray_dir[0] * e2[2],
        ray_dir[0] * e2[1] - ray_dir[1] * e2[0]
    };

    float a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if (fabsf(a) < EPSILON) return false;

    float f = 1.0f / a;

    // Vector from v0 to ray origin
    float s[3] = {
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2]
    };

    u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0f || u > 1.0f) return false;

    // Cross product s × e1
    float q[3] = {
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0]
    };

    v = f * (ray_dir[0] * q[0] + ray_dir[1] * q[1] + ray_dir[2] * q[2]);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);

    return t > EPSILON;
}

/**
 * @brief Compute signed volume of tetrahedron
 */
__device__ __forceinline__ float tet_volume(
    const float* a, const float* b, const float* c, const float* d) {

    float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    float ad[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};

    return (ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) -
            ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) +
            ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])) / 6.0f;
}

/**
 * @brief Main ray-tetrahedron intersection kernel
 */
__global__ void ray_tet_intersection_kernel(
    const float* __restrict__ ray_origins,     // [N, 3]
    const float* __restrict__ ray_dirs,        // [N, 3]
    const float* __restrict__ vertices,        // [V, 3]
    const int64_t* __restrict__ tetrahedra,    // [T, 4]
    uint8_t* __restrict__ hit_mask,            // [N, T]
    float* __restrict__ barycentrics,          // [N, T, 4]
    float* __restrict__ depths,                // [N, T]
    size_t N,
    size_t T) {

    size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tet_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (ray_idx >= N || tet_idx >= T) return;

    // Get ray
    float ray_o[3] = {
        ray_origins[ray_idx * 3 + 0],
        ray_origins[ray_idx * 3 + 1],
        ray_origins[ray_idx * 3 + 2]
    };
    float ray_d[3] = {
        ray_dirs[ray_idx * 3 + 0],
        ray_dirs[ray_idx * 3 + 1],
        ray_dirs[ray_idx * 3 + 2]
    };

    // Get tetrahedron vertices
    int64_t vi[4];
    float v[4][3];
    for (int i = 0; i < 4; ++i) {
        vi[i] = tetrahedra[tet_idx * 4 + i];
        v[i][0] = vertices[vi[i] * 3 + 0];
        v[i][1] = vertices[vi[i] * 3 + 1];
        v[i][2] = vertices[vi[i] * 3 + 2];
    }

    // Test intersection with each face
    // Face ordering: (1,2,3), (0,3,2), (0,1,3), (0,2,1)
    constexpr int faces[4][3] = {
        {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
    };

    size_t out_idx = ray_idx * T + tet_idx;
    hit_mask[out_idx] = 0;
    depths[out_idx] = 1e10f;

    for (int f = 0; f < 4; ++f) {
        float t, u, w;
        if (ray_triangle_intersect(ray_o, ray_d,
                                    v[faces[f][0]], v[faces[f][1]], v[faces[f][2]],
                                    t, u, w)) {

            // Compute hit point
            float hit[3] = {
                ray_o[0] + t * ray_d[0],
                ray_o[1] + t * ray_d[1],
                ray_o[2] + t * ray_d[2]
            };

            // Compute barycentric coordinates for the full tetrahedron
            float vol_total = fabsf(tet_volume(v[0], v[1], v[2], v[3]));
            if (vol_total < 1e-10f) continue;

            float bary[4];
            bary[0] = fabsf(tet_volume(hit, v[1], v[2], v[3])) / vol_total;
            bary[1] = fabsf(tet_volume(v[0], hit, v[2], v[3])) / vol_total;
            bary[2] = fabsf(tet_volume(v[0], v[1], hit, v[3])) / vol_total;
            bary[3] = fabsf(tet_volume(v[0], v[1], v[2], hit)) / vol_total;

            hit_mask[out_idx] = 1;
            depths[out_idx] = t;
            barycentrics[out_idx * 4 + 0] = bary[0];
            barycentrics[out_idx * 4 + 1] = bary[1];
            barycentrics[out_idx * 4 + 2] = bary[2];
            barycentrics[out_idx * 4 + 3] = bary[3];
            break;
        }
    }
}

/**
 * @brief Collect per-pixel intersections from hit_mask into compact arrays
 *
 * Scans through all tetrahedra for each pixel and collects up to max_K
 * intersections, storing tet indices, depths, and barycentric coordinates.
 * Results are written in arbitrary order; use sort_intersections after.
 */
__global__ void collect_intersections_kernel(
    const uint8_t* __restrict__ hit_mask,       // [N, T]
    const float* __restrict__ all_depths,       // [N, T]
    const float* __restrict__ all_barycentrics, // [N, T, 4]
    int64_t* __restrict__ out_tet_indices,      // [N, max_K]
    float* __restrict__ out_depths,             // [N, max_K]
    float* __restrict__ out_barycentrics,       // [N, max_K, 4]
    int32_t* __restrict__ out_num_intersects,   // [N]
    size_t N,
    size_t T,
    int max_K,
    float near_clip,
    float far_clip) {

    size_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= N) return;

    int count = 0;

    // Scan all tetrahedra for this ray
    for (size_t tet_idx = 0; tet_idx < T && count < max_K; ++tet_idx) {
        size_t in_idx = ray_idx * T + tet_idx;

        if (hit_mask[in_idx]) {
            float depth = all_depths[in_idx];

            // Apply clipping
            if (depth >= near_clip && depth <= far_clip) {
                size_t out_idx = ray_idx * max_K + count;

                out_tet_indices[out_idx] = static_cast<int64_t>(tet_idx);
                out_depths[out_idx] = depth;
                out_barycentrics[out_idx * 4 + 0] = all_barycentrics[in_idx * 4 + 0];
                out_barycentrics[out_idx * 4 + 1] = all_barycentrics[in_idx * 4 + 1];
                out_barycentrics[out_idx * 4 + 2] = all_barycentrics[in_idx * 4 + 2];
                out_barycentrics[out_idx * 4 + 3] = all_barycentrics[in_idx * 4 + 3];

                ++count;
            }
        }
    }

    // Initialize remaining slots to invalid
    for (int k = count; k < max_K; ++k) {
        size_t out_idx = ray_idx * max_K + k;
        out_tet_indices[out_idx] = -1;
        out_depths[out_idx] = 1e10f;
    }

    out_num_intersects[ray_idx] = count;
}

/**
 * @brief Tile-based ray-tet intersection with direct collection
 *
 * Processes rays in tiles, testing only tetrahedra that overlap the tile's
 * screen-space bounding box. This reduces the NxT complexity significantly
 * for spatially coherent scenes.
 *
 * Each thread handles one ray and iterates over candidate tetrahedra,
 * collecting up to max_K intersections directly.
 */
__global__ void ray_tet_intersection_tiled_kernel(
    const float* __restrict__ ray_origins,      // [N, 3]
    const float* __restrict__ ray_dirs,         // [N, 3]
    const float* __restrict__ vertices,         // [V, 3]
    const int64_t* __restrict__ tetrahedra,     // [T, 4]
    const float* __restrict__ tet_bounds,       // [T, 4] (min_x, min_y, max_x, max_y)
    int64_t* __restrict__ out_tet_indices,      // [N, max_K]
    float* __restrict__ out_depths,             // [N, max_K]
    float* __restrict__ out_barycentrics,       // [N, max_K, 4]
    int32_t* __restrict__ out_num_intersects,   // [N]
    int width,
    int height,
    size_t T,
    int max_K,
    float near_clip,
    float far_clip) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    size_t ray_idx = py * width + px;

    // Get ray for this pixel
    float ray_o[3] = {
        ray_origins[ray_idx * 3 + 0],
        ray_origins[ray_idx * 3 + 1],
        ray_origins[ray_idx * 3 + 2]
    };
    float ray_d[3] = {
        ray_dirs[ray_idx * 3 + 0],
        ray_dirs[ray_idx * 3 + 1],
        ray_dirs[ray_idx * 3 + 2]
    };

    // Temporary storage for intersections (heap-sort style)
    // We maintain a max-heap of max_K closest intersections
    float local_depths[64];     // Assuming max_K <= 64
    int64_t local_tets[64];
    float local_bary[64 * 4];
    int count = 0;

    // Test all tetrahedra (with bounding box culling)
    for (size_t tet_idx = 0; tet_idx < T; ++tet_idx) {
        // Check if pixel is within tet's screen-space bounds
        float min_x = tet_bounds[tet_idx * 4 + 0];
        float min_y = tet_bounds[tet_idx * 4 + 1];
        float max_x = tet_bounds[tet_idx * 4 + 2];
        float max_y = tet_bounds[tet_idx * 4 + 3];

        // Skip invalid bounds (off-screen tetrahedra)
        if (min_x < 0) continue;

        // Conservative bounds check with 1-pixel margin
        if (px < min_x - 1 || px > max_x + 1 ||
            py < min_y - 1 || py > max_y + 1) {
            continue;
        }

        // Get tetrahedron vertices
        int64_t vi[4];
        float v[4][3];
        for (int i = 0; i < 4; ++i) {
            vi[i] = tetrahedra[tet_idx * 4 + i];
            v[i][0] = vertices[vi[i] * 3 + 0];
            v[i][1] = vertices[vi[i] * 3 + 1];
            v[i][2] = vertices[vi[i] * 3 + 2];
        }

        // Test intersection with each face
        constexpr int faces[4][3] = {
            {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
        };

        for (int f = 0; f < 4; ++f) {
            float t, u, w;
            if (ray_triangle_intersect(ray_o, ray_d,
                                        v[faces[f][0]], v[faces[f][1]], v[faces[f][2]],
                                        t, u, w)) {

                // Apply clipping
                if (t < near_clip || t > far_clip) break;

                // Compute hit point
                float hit[3] = {
                    ray_o[0] + t * ray_d[0],
                    ray_o[1] + t * ray_d[1],
                    ray_o[2] + t * ray_d[2]
                };

                // Compute barycentric coordinates
                float vol_total = fabsf(tet_volume(v[0], v[1], v[2], v[3]));
                if (vol_total < 1e-10f) break;

                float bary[4];
                bary[0] = fabsf(tet_volume(hit, v[1], v[2], v[3])) / vol_total;
                bary[1] = fabsf(tet_volume(v[0], hit, v[2], v[3])) / vol_total;
                bary[2] = fabsf(tet_volume(v[0], v[1], hit, v[3])) / vol_total;
                bary[3] = fabsf(tet_volume(v[0], v[1], v[2], hit)) / vol_total;

                // Insert into local storage maintaining sorted order by depth
                if (count < max_K) {
                    // Find insertion point
                    int insert_pos = count;
                    for (int k = 0; k < count; ++k) {
                        if (t < local_depths[k]) {
                            insert_pos = k;
                            break;
                        }
                    }

                    // Shift elements
                    for (int k = count; k > insert_pos; --k) {
                        local_depths[k] = local_depths[k - 1];
                        local_tets[k] = local_tets[k - 1];
                        for (int b = 0; b < 4; ++b) {
                            local_bary[k * 4 + b] = local_bary[(k - 1) * 4 + b];
                        }
                    }

                    // Insert
                    local_depths[insert_pos] = t;
                    local_tets[insert_pos] = static_cast<int64_t>(tet_idx);
                    for (int b = 0; b < 4; ++b) {
                        local_bary[insert_pos * 4 + b] = bary[b];
                    }
                    ++count;
                } else if (t < local_depths[max_K - 1]) {
                    // Replace the farthest intersection if this is closer
                    int insert_pos = max_K - 1;
                    for (int k = 0; k < max_K - 1; ++k) {
                        if (t < local_depths[k]) {
                            insert_pos = k;
                            break;
                        }
                    }

                    // Shift elements (drop the last one)
                    for (int k = max_K - 1; k > insert_pos; --k) {
                        local_depths[k] = local_depths[k - 1];
                        local_tets[k] = local_tets[k - 1];
                        for (int b = 0; b < 4; ++b) {
                            local_bary[k * 4 + b] = local_bary[(k - 1) * 4 + b];
                        }
                    }

                    // Insert
                    local_depths[insert_pos] = t;
                    local_tets[insert_pos] = static_cast<int64_t>(tet_idx);
                    for (int b = 0; b < 4; ++b) {
                        local_bary[insert_pos * 4 + b] = bary[b];
                    }
                }

                break;  // One intersection per tet is enough
            }
        }
    }

    // Write results to global memory
    out_num_intersects[ray_idx] = count;

    for (int k = 0; k < max_K; ++k) {
        size_t out_idx = ray_idx * max_K + k;
        if (k < count) {
            out_tet_indices[out_idx] = local_tets[k];
            out_depths[out_idx] = local_depths[k];
            for (int b = 0; b < 4; ++b) {
                out_barycentrics[out_idx * 4 + b] = local_bary[k * 4 + b];
            }
        } else {
            out_tet_indices[out_idx] = -1;
            out_depths[out_idx] = 1e10f;
            for (int b = 0; b < 4; ++b) {
                out_barycentrics[out_idx * 4 + b] = 0.0f;
            }
        }
    }
}

/**
 * @brief Query per-tet linear color model at intersection points
 *
 * For each valid intersection, computes the 3D position from barycentric
 * coordinates and evaluates the linear color model:
 *   color = base_color + dot(gradient, position)
 *   alpha = 1 - exp(-exp(density) * segment_length)
 *
 * This matches the radiance_meshes Python implementation.
 */
__global__ void compute_intersection_colors_kernel(
    const float* __restrict__ vertices,         // [V, 3]
    const int64_t* __restrict__ tetrahedra,     // [T, 4]
    const int64_t* __restrict__ tet_indices,    // [H*W, max_K]
    const float* __restrict__ barycentrics,     // [H*W, max_K, 4]
    const float* __restrict__ depths,           // [H*W, max_K]
    const int32_t* __restrict__ num_intersects, // [H*W]
    const float* __restrict__ tet_density,      // [T] log-scale density per tet
    const float* __restrict__ tet_base_color,   // [T, 3] base color per tet (or network RGB)
    const float* __restrict__ tet_gradient,     // [T, 3] color gradient per tet
    float* __restrict__ out_rgb,                // [H*W, max_K, 3]
    float* __restrict__ out_alpha,              // [H*W, max_K]
    int width,
    int height,
    int max_K,
    bool use_network_colors) {  // If true, use base_color directly as RGB (no softplus)

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    for (int k = 0; k < max_K; ++k) {
        int idx = pixel_idx * max_K + k;

        if (k < num_k) {
            int64_t tet_idx = tet_indices[idx];

            // Bounds check - skip if tet_idx is invalid
            if (tet_idx < 0) {
                out_rgb[idx * 3 + 0] = 0.0f;
                out_rgb[idx * 3 + 1] = 0.0f;
                out_rgb[idx * 3 + 2] = 0.0f;
                out_alpha[idx] = 0.0f;
                continue;
            }

            // Get tetrahedron vertices
            float v[4][3];
            for (int i = 0; i < 4; ++i) {
                int64_t vi = tetrahedra[tet_idx * 4 + i];
                v[i][0] = vertices[vi * 3 + 0];
                v[i][1] = vertices[vi * 3 + 1];
                v[i][2] = vertices[vi * 3 + 2];
            }

            // Compute 3D position from barycentric coordinates
            float bary0 = barycentrics[idx * 4 + 0];
            float bary1 = barycentrics[idx * 4 + 1];
            float bary2 = barycentrics[idx * 4 + 2];
            float bary3 = barycentrics[idx * 4 + 3];

            float pos_x = bary0 * v[0][0] + bary1 * v[1][0] + bary2 * v[2][0] + bary3 * v[3][0];
            float pos_y = bary0 * v[0][1] + bary1 * v[1][1] + bary2 * v[2][1] + bary3 * v[3][1];
            float pos_z = bary0 * v[0][2] + bary1 * v[1][2] + bary2 * v[2][2] + bary3 * v[3][2];

            // Per-tet linear color model (matches radiance_meshes Python):
            // color = softplus(base_color + dot(gradient, position), beta=10)
            // alpha = 1 - exp(-exp(density) * segment_length)
            // The gradient is a SINGLE 3D vector applied to ALL color channels

            // Get per-tet parameters
            float base_r = tet_base_color[tet_idx * 3 + 0];
            float base_g = tet_base_color[tet_idx * 3 + 1];
            float base_b = tet_base_color[tet_idx * 3 + 2];

            // Clamp density to prevent numerical overflow in exp()
            float density = fmaxf(-10.0f, fminf(10.0f, tet_density[tet_idx]));

            float r, g, b;

            if (use_network_colors) {
                // Network colors: base_color IS the final RGB (already sigmoid-activated)
                // No softplus, no gradient - use directly
                r = base_r;
                g = base_g;
                b = base_b;
            } else {
                // Per-tet linear color model (matches radiance_meshes Python):
                // color = softplus(base_color + dot(gradient, position), beta=10)
                // Gradient is a 3D direction vector (same for all color channels)
                float grad_x = tet_gradient[tet_idx * 3 + 0];
                float grad_y = tet_gradient[tet_idx * 3 + 1];
                float grad_z = tet_gradient[tet_idx * 3 + 2];

                // Linear color model: dot product of gradient with position
                // This offset is the SAME for all color channels (key insight!)
                float grad_offset = grad_x * pos_x + grad_y * pos_y + grad_z * pos_z;
                grad_offset = fmaxf(-20.0f, fminf(20.0f, grad_offset));

                // Softplus activation: softplus(x, beta) = log(1 + exp(beta*x)) / beta
                // With beta=10 for sharper activation (matches Python)
                const float beta = 10.0f;
                float input_r = base_r + grad_offset;
                float input_g = base_g + grad_offset;
                float input_b = base_b + grad_offset;

                // Numerically stable softplus: log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
                auto softplus = [beta](float x) -> float {
                    float bx = beta * x;
                    if (bx > 20.0f) return x;  // Avoid overflow
                    if (bx < -20.0f) return 0.0f;
                    return logf(1.0f + expf(bx)) / beta;
                };

                r = softplus(input_r);
                g = softplus(input_g);
                b = softplus(input_b);
            }

            // Clamp to [0, 1] for valid RGB
            r = fminf(1.0f, fmaxf(0.0f, r));
            g = fminf(1.0f, fmaxf(0.0f, g));
            b = fminf(1.0f, fmaxf(0.0f, b));

            // Compute segment length (distance to next intersection or default)
            float segment_length = 0.1f;  // Default for last intersection
            if (k + 1 < num_k) {
                int next_idx = pixel_idx * max_K + k + 1;
                float next_depth = depths[next_idx];
                float curr_depth = depths[idx];
                segment_length = fmaxf(next_depth - curr_depth, 0.001f);
            }

            // Volume rendering alpha: alpha = 1 - exp(-sigma * L)
            // where sigma = exp(density) to ensure positivity
            float sigma = expf(density);  // density already clamped above
            float alpha = 1.0f - expf(-sigma * segment_length);

            out_rgb[idx * 3 + 0] = r;
            out_rgb[idx * 3 + 1] = g;
            out_rgb[idx * 3 + 2] = b;
            out_alpha[idx] = alpha;
        } else {
            out_rgb[idx * 3 + 0] = 0.0f;
            out_rgb[idx * 3 + 1] = 0.0f;
            out_rgb[idx * 3 + 2] = 0.0f;
            out_alpha[idx] = 0.0f;
        }
    }
}

/**
 * @brief Compute first-hit depth buffer from sorted intersections
 */
__global__ void compute_depth_buffer_kernel(
    const float* __restrict__ depths,           // [H*W, max_K]
    const int32_t* __restrict__ num_intersects, // [H*W]
    float* __restrict__ depth_buffer,           // [H*W]
    int width,
    int height,
    int max_K) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    if (num_k > 0) {
        depth_buffer[pixel_idx] = depths[pixel_idx * max_K];  // First (closest) intersection
    } else {
        depth_buffer[pixel_idx] = 0.0f;
    }
}

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

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
    void* stream) {

    // 2D grid: rays x tetrahedra
    dim3 block_size(16, 16);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (T + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    ray_tet_intersection_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_dirs, vertices, tetrahedra,
        hit_mask, barycentrics, depths, N, T);
}

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
    void* stream) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    collect_intersections_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        hit_mask, all_depths, all_barycentrics,
        out_tet_indices, out_depths, out_barycentrics, out_num_intersects,
        N, T, max_K, near_clip, far_clip);
}

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    ray_tet_intersection_tiled_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_dirs, vertices, tetrahedra, tet_bounds,
        out_tet_indices, out_depths, out_barycentrics, out_num_intersects,
        width, height, T, max_K, near_clip, far_clip);
}

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
    bool use_network_colors,
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_intersection_colors_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, tetrahedra, tet_indices, barycentrics, depths, num_intersects,
        tet_density, tet_base_color, tet_gradient,
        out_rgb, out_alpha, width, height, max_K, use_network_colors);
}

void launch_compute_depth_buffer(
    const float* depths,
    const int32_t* num_intersects,
    float* depth_buffer,
    int width,
    int height,
    int max_K,
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_depth_buffer_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        depths, num_intersects, depth_buffer, width, height, max_K);
}

// ==============================================================================
// TETRAHEDRON CENTROID COMPUTATION
// ==============================================================================

/**
 * @brief Compute centroids for all tetrahedra
 *
 * centroid[t] = (v[t,0] + v[t,1] + v[t,2] + v[t,3]) / 4
 */
__global__ void compute_tet_centroids_kernel(
    const float* __restrict__ vertices,     // [V, 3]
    const int64_t* __restrict__ tetrahedra, // [T, 4]
    float* __restrict__ centroids,          // [T, 3]
    size_t num_tets) {

    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tets) return;

    // Get vertex indices for this tet
    int64_t v0 = tetrahedra[t * 4 + 0];
    int64_t v1 = tetrahedra[t * 4 + 1];
    int64_t v2 = tetrahedra[t * 4 + 2];
    int64_t v3 = tetrahedra[t * 4 + 3];

    // Compute centroid
    centroids[t * 3 + 0] = (vertices[v0 * 3 + 0] + vertices[v1 * 3 + 0] +
                           vertices[v2 * 3 + 0] + vertices[v3 * 3 + 0]) * 0.25f;
    centroids[t * 3 + 1] = (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1] +
                           vertices[v2 * 3 + 1] + vertices[v3 * 3 + 1]) * 0.25f;
    centroids[t * 3 + 2] = (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2] +
                           vertices[v2 * 3 + 2] + vertices[v3 * 3 + 2]) * 0.25f;
}

void launch_compute_tet_centroids(
    const float* vertices,
    const int64_t* tetrahedra,
    float* centroids,
    size_t num_tets,
    void* stream) {

    const int block_size = 256;
    const int grid_size = (num_tets + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_tet_centroids_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, tetrahedra, centroids, num_tets);
}

// ==============================================================================
// BACKWARD PASS FOR PER-TETRAHEDRON COLOR PARAMETERS
// ==============================================================================

namespace {

/**
 * @brief Backward kernel for per-tet linear color model
 *
 * Computes gradients for tetrahedron parameters given upstream gradients.
 *
 * Forward pass (for reference):
 *   position = sum(bary[i] * vertex[i])
 *   color = base_color + dot(gradient, position)
 *   sigma = exp(density)
 *   alpha = 1 - exp(-sigma * segment_length)
 *
 * Gradient derivations:
 *   d_loss/d_base_color = grad_rgb (since d_color/d_base_color = 1)
 *   d_loss/d_gradient = grad_rgb * position (since d_color/d_gradient = position)
 *   d_loss/d_density = grad_alpha * L * sigma * (1 - alpha)
 *     where L = segment_length, sigma = exp(density)
 */
__global__ void compute_intersection_colors_backward_kernel(
    // Forward inputs (for recomputation)
    const float* __restrict__ vertices,         // [V, 3]
    const int64_t* __restrict__ tetrahedra,     // [T, 4]
    const int64_t* __restrict__ tet_indices,    // [H*W, max_K]
    const float* __restrict__ barycentrics,     // [H*W, max_K, 4]
    const float* __restrict__ depths,           // [H*W, max_K]
    const int32_t* __restrict__ num_intersects, // [H*W]
    const float* __restrict__ tet_density,      // [T]
    const float* __restrict__ tet_base_color,   // [T, 3]
    const float* __restrict__ tet_gradient,     // [T, 3]

    // Upstream gradients (from alpha_blend_backward)
    const float* __restrict__ grad_rgb,         // [H*W, max_K, 3]
    const float* __restrict__ grad_alpha,       // [H*W, max_K]

    // Output gradients (use atomicAdd since multiple pixels may hit same tet)
    float* __restrict__ grad_density,           // [T]
    float* __restrict__ grad_base_color,        // [T, 3]
    float* __restrict__ grad_gradient,          // [T, 3]

    int width,
    int height,
    int max_K,
    int64_t num_tets,       // For bounds checking
    int64_t num_vertices) { // For bounds checking

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    for (int k = 0; k < max_K; ++k) {
        int idx = pixel_idx * max_K + k;

        if (k < num_k) {
            int64_t tet_idx = tet_indices[idx];

            // Skip invalid tetrahedra (includes out-of-bounds after re-triangulation)
            if (tet_idx < 0 || tet_idx >= num_tets) continue;

            // Get tetrahedron vertices with bounds checking
            float v[4][3];
            bool valid_vertices = true;
            for (int i = 0; i < 4; ++i) {
                int64_t vi = tetrahedra[tet_idx * 4 + i];
                if (vi < 0 || vi >= num_vertices) {
                    valid_vertices = false;
                    break;
                }
                v[i][0] = vertices[vi * 3 + 0];
                v[i][1] = vertices[vi * 3 + 1];
                v[i][2] = vertices[vi * 3 + 2];
            }
            if (!valid_vertices) continue;

            // Compute 3D position from barycentric coordinates
            float bary0 = barycentrics[idx * 4 + 0];
            float bary1 = barycentrics[idx * 4 + 1];
            float bary2 = barycentrics[idx * 4 + 2];
            float bary3 = barycentrics[idx * 4 + 3];

            float pos_x = bary0 * v[0][0] + bary1 * v[1][0] + bary2 * v[2][0] + bary3 * v[3][0];
            float pos_y = bary0 * v[0][1] + bary1 * v[1][1] + bary2 * v[2][1] + bary3 * v[3][1];
            float pos_z = bary0 * v[0][2] + bary1 * v[1][2] + bary2 * v[2][2] + bary3 * v[3][2];

            // Compute segment length (distance to next intersection or default)
            float segment_length = 0.1f;  // Default for last intersection
            if (k + 1 < num_k) {
                int next_idx = pixel_idx * max_K + k + 1;
                float next_depth = depths[next_idx];
                float curr_depth = depths[idx];
                segment_length = fmaxf(next_depth - curr_depth, 0.001f);
            }

            // Recompute forward values needed for backward
            // Must match forward kernel exactly (including clamping)
            float base_r = tet_base_color[tet_idx * 3 + 0];
            float base_g = tet_base_color[tet_idx * 3 + 1];
            float base_b = tet_base_color[tet_idx * 3 + 2];

            // Gradient is a 3D direction vector (same for all color channels)
            float grad_x = tet_gradient[tet_idx * 3 + 0];
            float grad_y = tet_gradient[tet_idx * 3 + 1];
            float grad_z = tet_gradient[tet_idx * 3 + 2];

            // Clamp density to prevent overflow (must match forward)
            float density = fmaxf(-10.0f, fminf(10.0f, tet_density[tet_idx]));

            // Recompute grad_offset (same for all channels)
            float grad_offset = grad_x * pos_x + grad_y * pos_y + grad_z * pos_z;
            grad_offset = fmaxf(-20.0f, fminf(20.0f, grad_offset));

            float input_r = base_r + grad_offset;
            float input_g = base_g + grad_offset;
            float input_b = base_b + grad_offset;

            // Softplus and its derivative
            // softplus(x, beta) = log(1 + exp(beta*x)) / beta
            // d_softplus/d_x = exp(beta*x) / (1 + exp(beta*x)) = sigmoid(beta*x)
            const float beta = 10.0f;
            auto softplus_deriv = [beta](float x) -> float {
                float bx = beta * x;
                if (bx > 20.0f) return 1.0f;
                if (bx < -20.0f) return 0.0f;
                float exp_bx = expf(bx);
                return exp_bx / (1.0f + exp_bx);  // sigmoid(beta*x)
            };

            float dsoftplus_r = softplus_deriv(input_r);
            float dsoftplus_g = softplus_deriv(input_g);
            float dsoftplus_b = softplus_deriv(input_b);

            float sigma = expf(density);
            float alpha = 1.0f - expf(-sigma * segment_length);

            // Get upstream gradients for this sample
            float g_r = grad_rgb[idx * 3 + 0];
            float g_g = grad_rgb[idx * 3 + 1];
            float g_b = grad_rgb[idx * 3 + 2];
            float g_alpha = grad_alpha[idx];

            // --------------------------------------------------------------
            // Softplus derivative: d_softplus/d_x = sigmoid(beta*x)
            // For color = softplus(base + offset):
            //   d_loss/d_base = grad_rgb * softplus'(input)
            //   d_loss/d_offset = grad_rgb * softplus'(input) (same for all channels)
            // --------------------------------------------------------------

            // Gradient for base_color (each channel independent)
            atomicAdd(&grad_base_color[tet_idx * 3 + 0], g_r * dsoftplus_r);
            atomicAdd(&grad_base_color[tet_idx * 3 + 1], g_g * dsoftplus_g);
            atomicAdd(&grad_base_color[tet_idx * 3 + 2], g_b * dsoftplus_b);

            // --------------------------------------------------------------
            // Gradient for spatial gradient vector:
            // grad_offset = dot(gradient, position)
            // d_loss/d_grad_offset = sum over channels of (grad_rgb * dsoftplus)
            // d_grad_offset/d_gradient = position
            // d_loss/d_gradient = d_loss/d_grad_offset * position
            // --------------------------------------------------------------
            float d_grad_offset = g_r * dsoftplus_r + g_g * dsoftplus_g + g_b * dsoftplus_b;
            atomicAdd(&grad_gradient[tet_idx * 3 + 0], d_grad_offset * pos_x);
            atomicAdd(&grad_gradient[tet_idx * 3 + 1], d_grad_offset * pos_y);
            atomicAdd(&grad_gradient[tet_idx * 3 + 2], d_grad_offset * pos_z);

            // --------------------------------------------------------------
            // Gradient for density:
            // alpha = 1 - exp(-sigma * L), sigma = exp(density)
            // d_alpha/d_sigma = L * exp(-sigma * L) = L * (1 - alpha)
            // d_sigma/d_density = sigma
            // d_alpha/d_density = d_alpha/d_sigma * d_sigma/d_density
            //                   = L * (1 - alpha) * sigma
            //                   = L * sigma * (1 - alpha)
            // d_loss/d_density = grad_alpha * d_alpha/d_density
            // --------------------------------------------------------------
            float d_alpha_d_density = segment_length * sigma * (1.0f - alpha);
            atomicAdd(&grad_density[tet_idx], g_alpha * d_alpha_d_density);
        }
    }
}

} // namespace

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_intersection_colors_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, tetrahedra, tet_indices, barycentrics, depths, num_intersects,
        tet_density, tet_base_color, tet_gradient,
        grad_rgb, grad_alpha,
        grad_density, grad_base_color, grad_gradient,
        width, height, max_K, num_tets, num_vertices);
}

// ==============================================================================
// TILE-BASED RASTERIZATION (3DGS-style for tetrahedra)
// ==============================================================================

/**
 * @brief Count how many tiles each tetrahedron touches
 *
 * First pass of tile-based rasterization. Uses screen-space bounding boxes
 * to determine which tiles each tet overlaps.
 */
__global__ void count_tiles_per_tet_kernel(
    const float* __restrict__ tet_bounds,    // [T, 4] min_x, min_y, max_x, max_y
    int32_t* __restrict__ tiles_per_tet,     // [T] output: tile count
    size_t num_tets,
    int tile_size,
    int tile_width,
    int tile_height,
    int image_width,
    int image_height) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets) return;

    float min_x = tet_bounds[idx * 4 + 0];
    float min_y = tet_bounds[idx * 4 + 1];
    float max_x = tet_bounds[idx * 4 + 2];
    float max_y = tet_bounds[idx * 4 + 3];

    // Invalid tet (off-screen or behind camera)
    if (min_x < 0) {
        tiles_per_tet[idx] = 0;
        return;
    }

    // Clamp bounds to screen (crucial for exterior tetrahedra)
    min_x = fmaxf(0.0f, min_x);
    min_y = fmaxf(0.0f, min_y);
    max_x = fminf(static_cast<float>(image_width), max_x);
    max_y = fminf(static_cast<float>(image_height), max_y);

    // Skip if completely outside screen after clamping
    if (min_x >= max_x || min_y >= max_y) {
        tiles_per_tet[idx] = 0;
        return;
    }

    // Skip tetrahedra with huge screen-space extent (likely exterior tets)
    // If a tet covers more than 30% of screen in EITHER dimension, skip it
    // This aggressively filters large exterior tetrahedra
    float extent_x = max_x - min_x;
    float extent_y = max_y - min_y;
    if (extent_x > image_width * 0.3f || extent_y > image_height * 0.3f) {
        tiles_per_tet[idx] = 0;
        return;
    }

    // Convert to tile coordinates
    int tile_min_x = max(0, static_cast<int>(floorf(min_x / tile_size)));
    int tile_min_y = max(0, static_cast<int>(floorf(min_y / tile_size)));
    int tile_max_x = min(tile_width, static_cast<int>(ceilf(max_x / tile_size)));
    int tile_max_y = min(tile_height, static_cast<int>(ceilf(max_y / tile_size)));

    int count = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
    tiles_per_tet[idx] = max(0, count);
}

/**
 * @brief Generate (tile_id, tet_idx) pairs for sorting
 *
 * Second pass of tile-based rasterization. Generates keys for radix sort.
 * Key format: (tile_id << 32) | depth_as_uint32
 */
__global__ void generate_tet_tile_pairs_kernel(
    const float* __restrict__ tet_bounds,    // [T, 4]
    const float* __restrict__ tet_depths,    // [T] centroid depth for sorting
    const int64_t* __restrict__ cum_tiles,   // [T] prefix sum of tiles_per_tet
    int64_t* __restrict__ isect_keys,        // [total_pairs] output: keys for sorting
    int32_t* __restrict__ tet_indices,       // [total_pairs] output: tet indices
    size_t num_tets,
    int tile_size,
    int tile_width,
    int tile_height) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets) return;

    float min_x = tet_bounds[idx * 4 + 0];
    float min_y = tet_bounds[idx * 4 + 1];
    float max_x = tet_bounds[idx * 4 + 2];
    float max_y = tet_bounds[idx * 4 + 3];

    // Invalid tet
    if (min_x < 0) return;

    // Convert to tile coordinates
    int tile_min_x = max(0, static_cast<int>(floorf(min_x / tile_size)));
    int tile_min_y = max(0, static_cast<int>(floorf(min_y / tile_size)));
    int tile_max_x = min(tile_width, static_cast<int>(ceilf(max_x / tile_size)));
    int tile_max_y = min(tile_height, static_cast<int>(ceilf(max_y / tile_size)));

    // Get starting offset from prefix sum
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles[idx - 1];

    // Encode depth as uint32 for stable sorting
    float depth = tet_depths[idx];
    uint32_t depth_bits = __float_as_uint(depth);

    // Generate pairs for each tile this tet touches
    for (int ty = tile_min_y; ty < tile_max_y; ++ty) {
        for (int tx = tile_min_x; tx < tile_max_x; ++tx) {
            int64_t tile_id = ty * tile_width + tx;
            // Key: tile_id in high 32 bits, depth in low 32 bits
            isect_keys[cur_idx] = (tile_id << 32) | depth_bits;
            tet_indices[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

/**
 * @brief Compute tile ranges from sorted keys
 *
 * After sorting by (tile_id, depth), find where each tile's data starts/ends.
 */
__global__ void compute_tile_ranges_kernel(
    const int64_t* __restrict__ sorted_keys,  // [total_pairs] sorted keys
    int32_t* __restrict__ tile_ranges,        // [num_tiles, 2] output: [start, end) per tile
    int64_t total_pairs,
    int num_tiles) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t tile_id_curr = sorted_keys[idx] >> 32;

    // First element initializes all tiles up to current
    if (idx == 0) {
        for (int i = 0; i <= tile_id_curr && i < num_tiles; ++i) {
            tile_ranges[i * 2 + 0] = 0;  // start
        }
    }

    // Last element finalizes all tiles from current onwards
    if (idx == total_pairs - 1) {
        for (int i = tile_id_curr; i < num_tiles; ++i) {
            tile_ranges[i * 2 + 1] = static_cast<int32_t>(total_pairs);  // end
        }
    }

    // Detect boundaries between tiles
    if (idx > 0) {
        int64_t tile_id_prev = sorted_keys[idx - 1] >> 32;
        if (tile_id_prev != tile_id_curr) {
            // End of previous tile's range
            tile_ranges[tile_id_prev * 2 + 1] = static_cast<int32_t>(idx);
            // Start of current tile's range
            for (int i = tile_id_prev + 1; i <= tile_id_curr && i < num_tiles; ++i) {
                tile_ranges[i * 2 + 0] = static_cast<int32_t>(idx);
                if (i < tile_id_curr) {
                    tile_ranges[i * 2 + 1] = static_cast<int32_t>(idx);  // empty tile
                }
            }
        }
    }
}

/**
 * @brief Tile-based ray-tet intersection using sorted tile lists
 *
 * FAST version that only tests tets assigned to each pixel's tile.
 * This reduces complexity from O(pixels × tets) to O(pixels × tets_per_tile).
 */
__global__ void ray_tet_intersection_with_tiles_kernel(
    const float* __restrict__ ray_origins,      // [H*W, 3]
    const float* __restrict__ ray_dirs,         // [H*W, 3]
    const float* __restrict__ vertices,         // [V, 3]
    const int64_t* __restrict__ tetrahedra,     // [T, 4]
    const int32_t* __restrict__ sorted_tet_ids, // [total_pairs] sorted tet indices
    const int32_t* __restrict__ tile_ranges,    // [num_tiles, 2] [start, end) per tile
    int64_t* __restrict__ out_tet_indices,      // [H*W, max_K]
    float* __restrict__ out_depths,             // [H*W, max_K]
    float* __restrict__ out_barycentrics,       // [H*W, max_K, 4]
    int32_t* __restrict__ out_num_intersects,   // [H*W]
    int width,
    int height,
    int tile_size,
    int tile_width,
    int max_K,
    float near_clip,
    float far_clip) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    size_t ray_idx = py * width + px;

    // Get ray for this pixel
    float ray_o[3] = {
        ray_origins[ray_idx * 3 + 0],
        ray_origins[ray_idx * 3 + 1],
        ray_origins[ray_idx * 3 + 2]
    };
    float ray_d[3] = {
        ray_dirs[ray_idx * 3 + 0],
        ray_dirs[ray_idx * 3 + 1],
        ray_dirs[ray_idx * 3 + 2]
    };

    // Determine which tile this pixel belongs to
    int tile_x = px / tile_size;
    int tile_y = py / tile_size;
    int tile_id = tile_y * tile_width + tile_x;

    // Get range of sorted tet indices for this tile
    int range_start = tile_ranges[tile_id * 2 + 0];
    int range_end = tile_ranges[tile_id * 2 + 1];

    // Local storage for intersections (sorted by depth)
    float local_depths[64];
    int64_t local_tets[64];
    float local_bary[64 * 4];
    int count = 0;

    // Test only tets that touch this tile
    for (int i = range_start; i < range_end && count < max_K; ++i) {
        int32_t tet_idx = sorted_tet_ids[i];

        // Get tetrahedron vertices
        int64_t vi[4];
        float v[4][3];
        for (int j = 0; j < 4; ++j) {
            vi[j] = tetrahedra[tet_idx * 4 + j];
            v[j][0] = vertices[vi[j] * 3 + 0];
            v[j][1] = vertices[vi[j] * 3 + 1];
            v[j][2] = vertices[vi[j] * 3 + 2];
        }

        // Test intersection with each face
        constexpr int faces[4][3] = {
            {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
        };

        for (int f = 0; f < 4; ++f) {
            float t, u, w;
            if (ray_triangle_intersect(ray_o, ray_d,
                                        v[faces[f][0]], v[faces[f][1]], v[faces[f][2]],
                                        t, u, w)) {

                // Apply clipping
                if (t < near_clip || t > far_clip) break;

                // Compute hit point
                float hit[3] = {
                    ray_o[0] + t * ray_d[0],
                    ray_o[1] + t * ray_d[1],
                    ray_o[2] + t * ray_d[2]
                };

                // Compute barycentric coordinates
                float vol_total = fabsf(tet_volume(v[0], v[1], v[2], v[3]));
                if (vol_total < 1e-10f) break;

                float bary[4];
                bary[0] = fabsf(tet_volume(hit, v[1], v[2], v[3])) / vol_total;
                bary[1] = fabsf(tet_volume(v[0], hit, v[2], v[3])) / vol_total;
                bary[2] = fabsf(tet_volume(v[0], v[1], hit, v[3])) / vol_total;
                bary[3] = fabsf(tet_volume(v[0], v[1], v[2], hit)) / vol_total;

                // Insert sorted by depth
                int insert_pos = count;
                for (int k = 0; k < count; ++k) {
                    if (t < local_depths[k]) {
                        insert_pos = k;
                        break;
                    }
                }

                // Shift elements
                for (int k = count; k > insert_pos; --k) {
                    local_depths[k] = local_depths[k - 1];
                    local_tets[k] = local_tets[k - 1];
                    for (int b = 0; b < 4; ++b) {
                        local_bary[k * 4 + b] = local_bary[(k - 1) * 4 + b];
                    }
                }

                // Insert
                local_depths[insert_pos] = t;
                local_tets[insert_pos] = static_cast<int64_t>(tet_idx);
                for (int b = 0; b < 4; ++b) {
                    local_bary[insert_pos * 4 + b] = bary[b];
                }
                ++count;

                break;  // One intersection per tet is enough
            }
        }
    }

    // Write results to global memory
    out_num_intersects[ray_idx] = count;

    for (int k = 0; k < max_K; ++k) {
        size_t out_idx = ray_idx * max_K + k;
        if (k < count) {
            out_tet_indices[out_idx] = local_tets[k];
            out_depths[out_idx] = local_depths[k];
            for (int b = 0; b < 4; ++b) {
                out_barycentrics[out_idx * 4 + b] = local_bary[k * 4 + b];
            }
        } else {
            out_tet_indices[out_idx] = -1;
            out_depths[out_idx] = 1e10f;
            for (int b = 0; b < 4; ++b) {
                out_barycentrics[out_idx * 4 + b] = 0.0f;
            }
        }
    }
}

/**
 * @brief Compute centroid depth for each tetrahedron
 */
__global__ void compute_tet_depths_kernel(
    const float* __restrict__ vertices,      // [V, 3]
    const int64_t* __restrict__ tetrahedra,  // [T, 4]
    const float* __restrict__ R,             // [3, 3] rotation
    const float* __restrict__ T_vec,         // [3] translation
    float* __restrict__ depths,              // [T] output: centroid depth
    size_t num_tets) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tets) return;

    // Compute centroid
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int i = 0; i < 4; ++i) {
        int64_t vi = tetrahedra[idx * 4 + i];
        cx += vertices[vi * 3 + 0];
        cy += vertices[vi * 3 + 1];
        cz += vertices[vi * 3 + 2];
    }
    cx *= 0.25f;
    cy *= 0.25f;
    cz *= 0.25f;

    // Transform to camera space and get depth (z component)
    float depth = R[6] * cx + R[7] * cy + R[8] * cz + T_vec[2];
    depths[idx] = depth;
}

// ==============================================================================
// PUBLIC API FOR TILE-BASED RASTERIZATION
// ==============================================================================

void launch_count_tiles_per_tet(
    const float* tet_bounds,
    int32_t* tiles_per_tet,
    size_t num_tets,
    int tile_size,
    int tile_width,
    int tile_height,
    int image_width,
    int image_height,
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_tets) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    count_tiles_per_tet_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        tet_bounds, tiles_per_tet, num_tets, tile_size, tile_width, tile_height,
        image_width, image_height);
}

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
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_tets) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    generate_tet_tile_pairs_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        tet_bounds, tet_depths, cum_tiles, isect_keys, tet_indices,
        num_tets, tile_size, tile_width, tile_height);
}

void launch_compute_tile_ranges(
    const int64_t* sorted_keys,
    int32_t* tile_ranges,
    int64_t total_pairs,
    int num_tiles,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    // CRITICAL: Initialize ALL tile ranges to empty (0, 0) first
    // This prevents uninitialized memory access for tiles with no tets
    cudaMemsetAsync(tile_ranges, 0, num_tiles * 2 * sizeof(int32_t), cuda_stream);

    if (total_pairs == 0) {
        return;
    }

    int block_size = 256;
    int grid_size = (static_cast<int>(total_pairs) + block_size - 1) / block_size;

    compute_tile_ranges_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        sorted_keys, tile_ranges, total_pairs, num_tiles);
}

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    ray_tet_intersection_with_tiles_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_dirs, vertices, tetrahedra, sorted_tet_ids, tile_ranges,
        out_tet_indices, out_depths, out_barycentrics, out_num_intersects,
        width, height, tile_size, tile_width, max_K, near_clip, far_clip);
}

void launch_compute_tet_depths(
    const float* vertices,
    const int64_t* tetrahedra,
    const float* R,
    const float* T_vec,
    float* depths,
    size_t num_tets,
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_tets) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_tet_depths_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, tetrahedra, R, T_vec, depths, num_tets);
}

// ==============================================================================
// SGD PARAMETER UPDATE (for neural network training)
// ==============================================================================

/**
 * @brief Simple SGD parameter update: param = param - lr * grad
 */
__global__ void sgd_update_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float lr,
    size_t n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    param[idx] -= lr * grad[idx];
}

void launch_sgd_update(
    float* param,
    const float* grad,
    float lr,
    size_t n,
    void* stream) {

    if (n == 0) return;

    int block_size = 256;
    int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    sgd_update_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        param, grad, lr, n);
}

} // namespace lfs::tetra::cuda
