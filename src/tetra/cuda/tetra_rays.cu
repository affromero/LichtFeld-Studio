/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tetra_rays.cu
 * @brief CUDA kernels for generating camera rays from intrinsic/extrinsic parameters
 *
 * Generates ray origins and directions for all pixels in parallel given camera
 * intrinsics (fx, fy, cx, cy) and extrinsics (R, T world-to-camera transform).
 */

#include <cuda_runtime.h>
#include <cmath>

namespace lfs::tetra::cuda {

namespace {

/**
 * @brief Generate rays from camera parameters
 *
 * For each pixel (px, py), computes:
 * 1. Ray direction in camera space: dir_cam = [(px - cx)/fx, (py - cy)/fy, 1.0]
 * 2. Transform to world space: dir_world = R^T @ dir_cam
 * 3. Normalize: dir_world /= |dir_world|
 * 4. Origin in world space: R^T @ (-T)
 *
 * The rotation matrix R is stored in row-major order as [r00, r01, r02, r10, ...].
 * R^T is computed by swapping indices during access.
 */
__global__ void generate_rays_kernel(
    float* __restrict__ ray_origins,      // [H*W, 3] output
    float* __restrict__ ray_directions,   // [H*W, 3] output
    const float* __restrict__ R,          // [3, 3] rotation matrix (row-major)
    const float* __restrict__ T,          // [3] translation
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;

    // ------------------------------------------------------------
    // Step 1: Compute ray direction in camera space
    // ------------------------------------------------------------
    float dir_cam_x = (static_cast<float>(px) - cx) / fx;
    float dir_cam_y = (static_cast<float>(py) - cy) / fy;
    float dir_cam_z = 1.0f;

    // ------------------------------------------------------------
    // Step 2: Transform direction to world space using R^T
    // R is row-major: R[i,j] = R[i*3 + j]
    // R^T[i,j] = R[j,i] = R[j*3 + i]
    // dir_world = R^T @ dir_cam
    // ------------------------------------------------------------
    float dir_world_x = R[0] * dir_cam_x + R[3] * dir_cam_y + R[6] * dir_cam_z;
    float dir_world_y = R[1] * dir_cam_x + R[4] * dir_cam_y + R[7] * dir_cam_z;
    float dir_world_z = R[2] * dir_cam_x + R[5] * dir_cam_y + R[8] * dir_cam_z;

    // ------------------------------------------------------------
    // Step 3: Normalize direction
    // ------------------------------------------------------------
    float inv_norm = rsqrtf(
        dir_world_x * dir_world_x +
        dir_world_y * dir_world_y +
        dir_world_z * dir_world_z);

    dir_world_x *= inv_norm;
    dir_world_y *= inv_norm;
    dir_world_z *= inv_norm;

    // ------------------------------------------------------------
    // Step 4: Compute ray origin in world space
    // origin = R^T @ (-T)
    // ------------------------------------------------------------
    float neg_T_x = -T[0];
    float neg_T_y = -T[1];
    float neg_T_z = -T[2];

    float origin_x = R[0] * neg_T_x + R[3] * neg_T_y + R[6] * neg_T_z;
    float origin_y = R[1] * neg_T_x + R[4] * neg_T_y + R[7] * neg_T_z;
    float origin_z = R[2] * neg_T_x + R[5] * neg_T_y + R[8] * neg_T_z;

    // ------------------------------------------------------------
    // Write outputs
    // ------------------------------------------------------------
    ray_origins[pixel_idx * 3 + 0] = origin_x;
    ray_origins[pixel_idx * 3 + 1] = origin_y;
    ray_origins[pixel_idx * 3 + 2] = origin_z;

    ray_directions[pixel_idx * 3 + 0] = dir_world_x;
    ray_directions[pixel_idx * 3 + 1] = dir_world_y;
    ray_directions[pixel_idx * 3 + 2] = dir_world_z;
}

/**
 * @brief Generate rays with precomputed camera origin (optimized version)
 *
 * When rendering multiple pixels from the same camera, the origin is identical
 * for all rays. This version takes a precomputed origin to avoid redundant
 * computation and memory writes.
 */
__global__ void generate_rays_shared_origin_kernel(
    float* __restrict__ ray_origins,      // [H*W, 3] output
    float* __restrict__ ray_directions,   // [H*W, 3] output
    const float* __restrict__ R,          // [3, 3] rotation matrix (row-major)
    float origin_x,
    float origin_y,
    float origin_z,
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;

    // Ray direction in camera space
    float dir_cam_x = (static_cast<float>(px) - cx) / fx;
    float dir_cam_y = (static_cast<float>(py) - cy) / fy;
    float dir_cam_z = 1.0f;

    // Transform to world space: dir_world = R^T @ dir_cam
    float dir_world_x = R[0] * dir_cam_x + R[3] * dir_cam_y + R[6] * dir_cam_z;
    float dir_world_y = R[1] * dir_cam_x + R[4] * dir_cam_y + R[7] * dir_cam_z;
    float dir_world_z = R[2] * dir_cam_x + R[5] * dir_cam_y + R[8] * dir_cam_z;

    // Normalize direction
    float inv_norm = rsqrtf(
        dir_world_x * dir_world_x +
        dir_world_y * dir_world_y +
        dir_world_z * dir_world_z);

    dir_world_x *= inv_norm;
    dir_world_y *= inv_norm;
    dir_world_z *= inv_norm;

    // Write outputs
    ray_origins[pixel_idx * 3 + 0] = origin_x;
    ray_origins[pixel_idx * 3 + 1] = origin_y;
    ray_origins[pixel_idx * 3 + 2] = origin_z;

    ray_directions[pixel_idx * 3 + 0] = dir_world_x;
    ray_directions[pixel_idx * 3 + 1] = dir_world_y;
    ray_directions[pixel_idx * 3 + 2] = dir_world_z;
}

/**
 * @brief Generate rays for a subset of pixels (batch processing)
 *
 * Generates rays only for specified pixel coordinates, useful for:
 * - Sparse sampling during training
 * - Ray marching with adaptive sampling
 * - Interactive rendering with progressive refinement
 */
__global__ void generate_rays_batch_kernel(
    float* __restrict__ ray_origins,      // [N, 3] output
    float* __restrict__ ray_directions,   // [N, 3] output
    const int* __restrict__ pixel_coords, // [N, 2] (px, py) pairs
    const float* __restrict__ R,          // [3, 3] rotation matrix (row-major)
    const float* __restrict__ T,          // [3] translation
    float fx,
    float fy,
    float cx,
    float cy,
    int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Get pixel coordinates
    int px = pixel_coords[idx * 2 + 0];
    int py = pixel_coords[idx * 2 + 1];

    // Ray direction in camera space
    float dir_cam_x = (static_cast<float>(px) - cx) / fx;
    float dir_cam_y = (static_cast<float>(py) - cy) / fy;
    float dir_cam_z = 1.0f;

    // Transform to world space: dir_world = R^T @ dir_cam
    float dir_world_x = R[0] * dir_cam_x + R[3] * dir_cam_y + R[6] * dir_cam_z;
    float dir_world_y = R[1] * dir_cam_x + R[4] * dir_cam_y + R[7] * dir_cam_z;
    float dir_world_z = R[2] * dir_cam_x + R[5] * dir_cam_y + R[8] * dir_cam_z;

    // Normalize direction
    float inv_norm = rsqrtf(
        dir_world_x * dir_world_x +
        dir_world_y * dir_world_y +
        dir_world_z * dir_world_z);

    dir_world_x *= inv_norm;
    dir_world_y *= inv_norm;
    dir_world_z *= inv_norm;

    // Compute ray origin: R^T @ (-T)
    float neg_T_x = -T[0];
    float neg_T_y = -T[1];
    float neg_T_z = -T[2];

    float origin_x = R[0] * neg_T_x + R[3] * neg_T_y + R[6] * neg_T_z;
    float origin_y = R[1] * neg_T_x + R[4] * neg_T_y + R[7] * neg_T_z;
    float origin_z = R[2] * neg_T_x + R[5] * neg_T_y + R[8] * neg_T_z;

    // Write outputs
    ray_origins[idx * 3 + 0] = origin_x;
    ray_origins[idx * 3 + 1] = origin_y;
    ray_origins[idx * 3 + 2] = origin_z;

    ray_directions[idx * 3 + 0] = dir_world_x;
    ray_directions[idx * 3 + 1] = dir_world_y;
    ray_directions[idx * 3 + 2] = dir_world_z;
}

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    generate_rays_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_directions, R, T, fx, fy, cx, cy, width, height);
}

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    generate_rays_shared_origin_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_directions, R,
        origin_x, origin_y, origin_z,
        fx, fy, cx, cy, width, height);
}

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
    void* stream) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    generate_rays_batch_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        ray_origins, ray_directions, pixel_coords, R, T, fx, fy, cx, cy, N);
}

} // namespace lfs::tetra::cuda
