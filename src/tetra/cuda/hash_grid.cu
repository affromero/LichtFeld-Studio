/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hash_grid.cu
 * @brief CUDA kernels for multi-resolution hash grid encoding
 *
 * Implements instant-NGP style hash encoding for neural feature representation.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace lfs::tetra::cuda {

namespace {

// Prime numbers for spatial hashing
constexpr uint32_t PRIME1 = 1u;
constexpr uint32_t PRIME2 = 2654435761u;
constexpr uint32_t PRIME3 = 805459861u;

/**
 * @brief Spatial hash function for 3D grid coordinates
 */
__device__ __forceinline__ uint32_t spatial_hash(int x, int y, int z) {
    return (static_cast<uint32_t>(x) * PRIME1) ^
           (static_cast<uint32_t>(y) * PRIME2) ^
           (static_cast<uint32_t>(z) * PRIME3);
}

/**
 * @brief Hash grid encoding forward kernel
 *
 * Encodes positions using multi-resolution hash tables with trilinear interpolation.
 */
__global__ void hash_grid_encode_kernel(
    const float* __restrict__ positions,     // [N, 3]
    const float* __restrict__ hash_table,    // [total_entries, F]
    const float* __restrict__ aabb,          // [6] min/max
    float* __restrict__ output,              // [N, L*F]
    const float* __restrict__ resolutions,   // [L]
    const size_t* __restrict__ offsets,      // [L+1]
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Get position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // Normalize to [0, 1] using AABB
    float min_x = aabb[0], min_y = aabb[1], min_z = aabb[2];
    float max_x = aabb[3], max_y = aabb[4], max_z = aabb[5];

    float scale_x = 1.0f / (max_x - min_x + 1e-8f);
    float scale_y = 1.0f / (max_y - min_y + 1e-8f);
    float scale_z = 1.0f / (max_z - min_z + 1e-8f);

    px = fminf(fmaxf((px - min_x) * scale_x, 0.0f), 1.0f - 1e-6f);
    py = fminf(fmaxf((py - min_y) * scale_y, 0.0f), 1.0f - 1e-6f);
    pz = fminf(fmaxf((pz - min_z) * scale_z, 0.0f), 1.0f - 1e-6f);

    // For each resolution level
    for (int l = 0; l < num_levels; ++l) {
        float res = resolutions[l];
        size_t offset = offsets[l];

        // Grid coordinates
        float fx = px * res;
        float fy = py * res;
        float fz = pz * res;

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int z0 = static_cast<int>(floorf(fz));

        // Trilinear weights
        float wx = fx - x0;
        float wy = fy - y0;
        float wz = fz - z0;

        // Initialize output for this level
        int out_base = idx * num_levels * features_per_level + l * features_per_level;
        for (int f = 0; f < features_per_level; ++f) {
            output[out_base + f] = 0.0f;
        }

        // 8 corner contributions
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int xi = x0 + dx;
                    int yi = y0 + dy;
                    int zi = z0 + dz;

                    // Hash to table index
                    uint32_t hash = spatial_hash(xi, yi, zi);
                    size_t entry_idx = offset + (hash % table_size);

                    // Trilinear weight
                    float w = (dx ? wx : 1.0f - wx) *
                              (dy ? wy : 1.0f - wy) *
                              (dz ? wz : 1.0f - wz);

                    // Accumulate features
                    for (int f = 0; f < features_per_level; ++f) {
                        output[out_base + f] += w * hash_table[entry_idx * features_per_level + f];
                    }
                }
            }
        }
    }
}

/**
 * @brief Hash grid encoding backward kernel
 *
 * Computes gradients w.r.t. hash table entries using atomic operations.
 */
__global__ void hash_grid_backward_kernel(
    const float* __restrict__ positions,     // [N, 3]
    const float* __restrict__ grad_output,   // [N, L*F]
    const float* __restrict__ aabb,          // [6]
    float* __restrict__ grad_hash_table,     // [total_entries, F]
    const float* __restrict__ resolutions,   // [L]
    const size_t* __restrict__ offsets,      // [L+1]
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Get position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // Normalize to [0, 1] using AABB
    float min_x = aabb[0], min_y = aabb[1], min_z = aabb[2];
    float max_x = aabb[3], max_y = aabb[4], max_z = aabb[5];

    float scale_x = 1.0f / (max_x - min_x + 1e-8f);
    float scale_y = 1.0f / (max_y - min_y + 1e-8f);
    float scale_z = 1.0f / (max_z - min_z + 1e-8f);

    px = fminf(fmaxf((px - min_x) * scale_x, 0.0f), 1.0f - 1e-6f);
    py = fminf(fmaxf((py - min_y) * scale_y, 0.0f), 1.0f - 1e-6f);
    pz = fminf(fmaxf((pz - min_z) * scale_z, 0.0f), 1.0f - 1e-6f);

    // For each resolution level
    for (int l = 0; l < num_levels; ++l) {
        float res = resolutions[l];
        size_t offset = offsets[l];

        // Grid coordinates
        float fx = px * res;
        float fy = py * res;
        float fz = pz * res;

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int z0 = static_cast<int>(floorf(fz));

        // Trilinear weights
        float wx = fx - x0;
        float wy = fy - y0;
        float wz = fz - z0;

        // Get gradient for this level
        int grad_base = idx * num_levels * features_per_level + l * features_per_level;

        // 8 corner contributions
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int xi = x0 + dx;
                    int yi = y0 + dy;
                    int zi = z0 + dz;

                    // Hash to table index
                    uint32_t hash = spatial_hash(xi, yi, zi);
                    size_t entry_idx = offset + (hash % table_size);

                    // Trilinear weight
                    float w = (dx ? wx : 1.0f - wx) *
                              (dy ? wy : 1.0f - wy) *
                              (dz ? wz : 1.0f - wz);

                    // Accumulate gradients using atomics
                    for (int f = 0; f < features_per_level; ++f) {
                        atomicAdd(&grad_hash_table[entry_idx * features_per_level + f],
                                  w * grad_output[grad_base + f]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Hash grid position gradient kernel
 *
 * Computes gradients w.r.t. input positions.
 */
__global__ void hash_grid_position_backward_kernel(
    const float* __restrict__ positions,     // [N, 3]
    const float* __restrict__ grad_output,   // [N, L*F]
    const float* __restrict__ hash_table,    // [total_entries, F]
    const float* __restrict__ aabb,          // [6]
    float* __restrict__ grad_positions,      // [N, 3]
    const float* __restrict__ resolutions,   // [L]
    const size_t* __restrict__ offsets,      // [L+1]
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Get position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // AABB
    float min_x = aabb[0], min_y = aabb[1], min_z = aabb[2];
    float max_x = aabb[3], max_y = aabb[4], max_z = aabb[5];

    float scale_x = 1.0f / (max_x - min_x + 1e-8f);
    float scale_y = 1.0f / (max_y - min_y + 1e-8f);
    float scale_z = 1.0f / (max_z - min_z + 1e-8f);

    // Normalized position
    float norm_px = fminf(fmaxf((px - min_x) * scale_x, 0.0f), 1.0f - 1e-6f);
    float norm_py = fminf(fmaxf((py - min_y) * scale_y, 0.0f), 1.0f - 1e-6f);
    float norm_pz = fminf(fmaxf((pz - min_z) * scale_z, 0.0f), 1.0f - 1e-6f);

    float grad_px = 0.0f, grad_py = 0.0f, grad_pz = 0.0f;

    // For each resolution level
    for (int l = 0; l < num_levels; ++l) {
        float res = resolutions[l];
        size_t offset = offsets[l];

        // Grid coordinates
        float fx = norm_px * res;
        float fy = norm_py * res;
        float fz = norm_pz * res;

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int z0 = static_cast<int>(floorf(fz));

        // Trilinear weights
        float wx = fx - x0;
        float wy = fy - y0;
        float wz = fz - z0;

        // Get gradient for this level
        int grad_base = idx * num_levels * features_per_level + l * features_per_level;

        // Compute weight gradients for each corner
        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    int xi = x0 + dx;
                    int yi = y0 + dy;
                    int zi = z0 + dz;

                    // Hash to table index
                    uint32_t hash = spatial_hash(xi, yi, zi);
                    size_t entry_idx = offset + (hash % table_size);

                    // Weight derivatives
                    float dwx = (dx ? 1.0f : -1.0f);
                    float dwy = (dy ? 1.0f : -1.0f);
                    float dwz = (dz ? 1.0f : -1.0f);

                    float wx_val = (dx ? wx : 1.0f - wx);
                    float wy_val = (dy ? wy : 1.0f - wy);
                    float wz_val = (dz ? wz : 1.0f - wz);

                    // Accumulate position gradients
                    for (int f = 0; f < features_per_level; ++f) {
                        float hash_val = hash_table[entry_idx * features_per_level + f];
                        float grad_out = grad_output[grad_base + f];

                        grad_px += grad_out * hash_val * dwx * wy_val * wz_val * res * scale_x;
                        grad_py += grad_out * hash_val * wx_val * dwy * wz_val * res * scale_y;
                        grad_pz += grad_out * hash_val * wx_val * wy_val * dwz * res * scale_z;
                    }
                }
            }
        }
    }

    grad_positions[idx * 3 + 0] = grad_px;
    grad_positions[idx * 3 + 1] = grad_py;
    grad_positions[idx * 3 + 2] = grad_pz;
}

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

void launch_hash_grid_encode(
    const float* positions,
    const float* hash_table,
    const float* aabb,
    float* output,
    const float* resolutions,
    const size_t* offsets,
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size,
    void* stream) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    hash_grid_encode_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        positions, hash_table, aabb, output,
        resolutions, offsets, N, num_levels, features_per_level, table_size);
}

void launch_hash_grid_backward(
    const float* positions,
    const float* grad_output,
    const float* aabb,
    float* grad_hash_table,
    const float* resolutions,
    const size_t* offsets,
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size,
    void* stream) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    hash_grid_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        positions, grad_output, aabb, grad_hash_table,
        resolutions, offsets, N, num_levels, features_per_level, table_size);
}

void launch_hash_grid_position_backward(
    const float* positions,
    const float* grad_output,
    const float* hash_table,
    const float* aabb,
    float* grad_positions,
    const float* resolutions,
    const size_t* offsets,
    size_t N,
    int num_levels,
    int features_per_level,
    size_t table_size,
    void* stream) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    hash_grid_position_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        positions, grad_output, hash_table, aabb, grad_positions,
        resolutions, offsets, N, num_levels, features_per_level, table_size);
}

} // namespace lfs::tetra::cuda
