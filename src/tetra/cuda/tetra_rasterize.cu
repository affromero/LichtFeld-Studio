/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tetra_rasterize.cu
 * @brief CUDA kernels for tile-based tetrahedral mesh rasterization
 *
 * Implements front-to-back alpha blending for tetrahedral rendering.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace lfs::tetra::cuda {

namespace {

/**
 * @brief Alpha blending kernel
 *
 * Blends sorted intersections front-to-back with early termination.
 */
__global__ void alpha_blend_kernel(
    const float* __restrict__ rgb,           // [H, W, max_K, 3]
    const float* __restrict__ alpha,         // [H, W, max_K]
    const float* __restrict__ depths,        // [H, W, max_K]
    const int32_t* __restrict__ num_intersects,  // [H, W]
    const float* __restrict__ background,    // [3] or [H, W, 3]
    float* __restrict__ output_rgb,          // [H, W, 3]
    float* __restrict__ output_alpha,        // [H, W, 1]
    int width,
    int height,
    int max_K,
    float alpha_threshold,
    bool background_is_image) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    float accum_rgb[3] = {0.0f, 0.0f, 0.0f};
    float accum_alpha = 0.0f;

    // Front-to-back blending
    for (int k = 0; k < num_k && accum_alpha < alpha_threshold; ++k) {
        int idx = pixel_idx * max_K + k;

        float sample_alpha = alpha[idx];
        float sample_rgb[3] = {
            rgb[idx * 3 + 0],
            rgb[idx * 3 + 1],
            rgb[idx * 3 + 2]
        };

        // Alpha compositing
        float w = sample_alpha * (1.0f - accum_alpha);
        accum_rgb[0] += w * sample_rgb[0];
        accum_rgb[1] += w * sample_rgb[1];
        accum_rgb[2] += w * sample_rgb[2];
        accum_alpha += w;
    }

    // Apply background
    float bg[3];
    if (background_is_image) {
        bg[0] = background[pixel_idx * 3 + 0];
        bg[1] = background[pixel_idx * 3 + 1];
        bg[2] = background[pixel_idx * 3 + 2];
    } else {
        bg[0] = background[0];
        bg[1] = background[1];
        bg[2] = background[2];
    }

    output_rgb[pixel_idx * 3 + 0] = accum_rgb[0] + (1.0f - accum_alpha) * bg[0];
    output_rgb[pixel_idx * 3 + 1] = accum_rgb[1] + (1.0f - accum_alpha) * bg[1];
    output_rgb[pixel_idx * 3 + 2] = accum_rgb[2] + (1.0f - accum_alpha) * bg[2];
    output_alpha[pixel_idx] = accum_alpha;
}

/**
 * @brief Backward pass for alpha blending
 */
__global__ void alpha_blend_backward_kernel(
    const float* __restrict__ rgb,           // [H, W, max_K, 3]
    const float* __restrict__ alpha,         // [H, W, max_K]
    const float* __restrict__ depths,        // [H, W, max_K]
    const int32_t* __restrict__ num_intersects,
    const float* __restrict__ grad_output_rgb,   // [H, W, 3]
    const float* __restrict__ grad_output_alpha, // [H, W, 1] (may be null)
    float* __restrict__ grad_rgb,            // [H, W, max_K, 3]
    float* __restrict__ grad_alpha,          // [H, W, max_K]
    int width,
    int height,
    int max_K,
    float alpha_threshold) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    float grad_out_rgb[3] = {
        grad_output_rgb[pixel_idx * 3 + 0],
        grad_output_rgb[pixel_idx * 3 + 1],
        grad_output_rgb[pixel_idx * 3 + 2]
    };

    // Recompute forward pass to get intermediate values
    float accum_alphas[256];  // Assuming max_K <= 256
    float accum_alpha = 0.0f;

    for (int k = 0; k < num_k && k < 256; ++k) {
        accum_alphas[k] = accum_alpha;
        int idx = pixel_idx * max_K + k;
        float sample_alpha = alpha[idx];
        accum_alpha += sample_alpha * (1.0f - accum_alpha);

        if (accum_alpha >= alpha_threshold) break;
    }

    // Backward pass (reverse order)
    float grad_accum_alpha = 0.0f;

    for (int k = min(num_k - 1, 255); k >= 0; --k) {
        int idx = pixel_idx * max_K + k;

        float sample_alpha = alpha[idx];
        float transmittance = 1.0f - accum_alphas[k];
        float w = sample_alpha * transmittance;

        // Gradient w.r.t. sample RGB
        grad_rgb[idx * 3 + 0] = w * grad_out_rgb[0];
        grad_rgb[idx * 3 + 1] = w * grad_out_rgb[1];
        grad_rgb[idx * 3 + 2] = w * grad_out_rgb[2];

        // Gradient w.r.t. sample alpha
        float sample_rgb[3] = {
            rgb[idx * 3 + 0],
            rgb[idx * 3 + 1],
            rgb[idx * 3 + 2]
        };

        grad_alpha[idx] = transmittance * (
            sample_rgb[0] * grad_out_rgb[0] +
            sample_rgb[1] * grad_out_rgb[1] +
            sample_rgb[2] * grad_out_rgb[2]
        );

        // Accumulate gradient through transmittance
        grad_accum_alpha -= sample_alpha * (
            sample_rgb[0] * grad_out_rgb[0] +
            sample_rgb[1] * grad_out_rgb[1] +
            sample_rgb[2] * grad_out_rgb[2]
        );

        grad_alpha[idx] += grad_accum_alpha;
    }
}

/**
 * @brief Depth sorting kernel for per-pixel intersections
 */
__global__ void sort_intersections_kernel(
    float* __restrict__ depths,              // [H, W, max_K]
    int64_t* __restrict__ tet_indices,       // [H, W, max_K]
    float* __restrict__ barycentrics,        // [H, W, max_K, 4]
    const int32_t* __restrict__ num_intersects,
    int width,
    int height,
    int max_K) {

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    if (num_k <= 1) return;

    // Simple insertion sort (good for small K)
    for (int i = 1; i < num_k; ++i) {
        int idx_i = pixel_idx * max_K + i;
        float depth_i = depths[idx_i];
        int64_t tet_i = tet_indices[idx_i];
        float bary_i[4];
        for (int b = 0; b < 4; ++b) {
            bary_i[b] = barycentrics[idx_i * 4 + b];
        }

        int j = i - 1;
        while (j >= 0) {
            int idx_j = pixel_idx * max_K + j;
            if (depths[idx_j] <= depth_i) break;

            // Shift
            int idx_j1 = pixel_idx * max_K + j + 1;
            depths[idx_j1] = depths[idx_j];
            tet_indices[idx_j1] = tet_indices[idx_j];
            for (int b = 0; b < 4; ++b) {
                barycentrics[idx_j1 * 4 + b] = barycentrics[idx_j * 4 + b];
            }
            --j;
        }

        // Insert
        int idx_insert = pixel_idx * max_K + j + 1;
        depths[idx_insert] = depth_i;
        tet_indices[idx_insert] = tet_i;
        for (int b = 0; b < 4; ++b) {
            barycentrics[idx_insert * 4 + b] = bary_i[b];
        }
    }
}

// ------------------------------
// VERTEX PROJECTION KERNEL
// ------------------------------

/**
 * @brief Project vertices from world space to screen space
 *
 * Computes: p_cam = R @ p_world + T, then projects to 2D
 */
__global__ void project_vertices_kernel(
    const float* __restrict__ vertices,  // [V, 3] world positions
    const float* __restrict__ R,         // [3, 3] rotation matrix
    const float* __restrict__ T,         // [3] translation vector
    const float* __restrict__ intrinsics, // [4] fx, fy, cx, cy
    float* __restrict__ projected,       // [V, 4] (x, y, depth, valid)
    size_t num_vertices,
    float near_clip) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // Load vertex
    float vx = vertices[idx * 3 + 0];
    float vy = vertices[idx * 3 + 1];
    float vz = vertices[idx * 3 + 2];

    // World to camera: p_cam = R @ p_world + T
    float px = R[0] * vx + R[1] * vy + R[2] * vz + T[0];
    float py = R[3] * vx + R[4] * vy + R[5] * vz + T[1];
    float pz = R[6] * vx + R[7] * vy + R[8] * vz + T[2];

    // Load intrinsics
    float fx = intrinsics[0];
    float fy = intrinsics[1];
    float cx = intrinsics[2];
    float cy = intrinsics[3];

    // Project to screen
    float inv_z = 1.0f / (pz + 1e-8f);
    projected[idx * 4 + 0] = fx * px * inv_z + cx;  // x
    projected[idx * 4 + 1] = fy * py * inv_z + cy;  // y
    projected[idx * 4 + 2] = pz;                    // depth
    projected[idx * 4 + 3] = (pz > near_clip) ? 1.0f : 0.0f;  // valid
}

// ------------------------------
// TILE BOUNDING BOX KERNEL
// ------------------------------

/**
 * @brief Compute screen-space bounding boxes for tetrahedra
 */
__global__ void compute_tet_bounds_kernel(
    const int64_t* __restrict__ tetrahedra,  // [T, 4] vertex indices
    const float* __restrict__ projected,     // [V, 4] projected vertices
    float* __restrict__ bounds,              // [T, 4] min_x, min_y, max_x, max_y
    size_t num_tets) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tets) return;

    float min_x = 1e30f, min_y = 1e30f;
    float max_x = -1e30f, max_y = -1e30f;
    bool valid = true;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t vi = tetrahedra[t * 4 + i];
        float x = projected[vi * 4 + 0];
        float y = projected[vi * 4 + 1];
        float v = projected[vi * 4 + 3];

        if (v < 0.5f) {
            valid = false;
            break;
        }

        min_x = fminf(min_x, x);
        min_y = fminf(min_y, y);
        max_x = fmaxf(max_x, x);
        max_y = fmaxf(max_y, y);
    }

    if (!valid) {
        bounds[t * 4 + 0] = -1.0f;
        bounds[t * 4 + 1] = -1.0f;
        bounds[t * 4 + 2] = -1.0f;
        bounds[t * 4 + 3] = -1.0f;
    } else {
        bounds[t * 4 + 0] = min_x;
        bounds[t * 4 + 1] = min_y;
        bounds[t * 4 + 2] = max_x;
        bounds[t * 4 + 3] = max_y;
    }
}

// ------------------------------
// GRADIENT ACCUMULATION KERNEL
// ------------------------------

/**
 * @brief Accumulate L2 norm of gradients per vertex
 */
__global__ void accumulate_gradient_norms_kernel(
    const float* __restrict__ gradients,    // [V, 3] gradient vectors
    float* __restrict__ accum,              // [V] accumulated norms
    size_t num_vertices) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    float gx = gradients[idx * 3 + 0];
    float gy = gradients[idx * 3 + 1];
    float gz = gradients[idx * 3 + 2];
    float norm = sqrtf(gx * gx + gy * gy + gz * gz);

    atomicAdd(&accum[idx], norm);
}

// ------------------------------
// INTERSECTION RECONSTRUCTION KERNEL
// ------------------------------

/**
 * @brief Reconstruct intersection positions from barycentric coordinates
 *
 * Also computes view directions from camera position.
 */
__global__ void reconstruct_intersections_kernel(
    const float* __restrict__ vertices,       // [V, 3] vertex positions
    const int64_t* __restrict__ tetrahedra,   // [T, 4] tet indices
    const float* __restrict__ barycentrics,   // [H*W*K, 4] barycentric coords
    const int64_t* __restrict__ tet_indices,  // [H*W*K] tet index per sample
    const int32_t* __restrict__ num_intersects, // [H*W] intersections per pixel
    const float* __restrict__ cam_pos,        // [3] camera position
    float* __restrict__ positions,            // [total, 3] output positions
    float* __restrict__ directions,           // [total, 3] output directions
    const int32_t* __restrict__ prefix_sum,   // [H*W] prefix sum of intersections
    int width,
    int height,
    int max_K) {

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    int num_k = num_intersects[pixel_idx];
    if (num_k == 0) return;

    int out_offset = (pixel_idx > 0) ? prefix_sum[pixel_idx - 1] : 0;

    for (int k = 0; k < num_k; ++k) {
        int sample_idx = pixel_idx * max_K + k;
        int64_t tet_idx = tet_indices[sample_idx];

        // Get tet vertex indices
        int64_t v0 = tetrahedra[tet_idx * 4 + 0];
        int64_t v1 = tetrahedra[tet_idx * 4 + 1];
        int64_t v2 = tetrahedra[tet_idx * 4 + 2];
        int64_t v3 = tetrahedra[tet_idx * 4 + 3];

        // Get barycentric coords
        float b0 = barycentrics[sample_idx * 4 + 0];
        float b1 = barycentrics[sample_idx * 4 + 1];
        float b2 = barycentrics[sample_idx * 4 + 2];
        float b3 = barycentrics[sample_idx * 4 + 3];

        // Interpolate position
        float px = b0 * vertices[v0 * 3 + 0] + b1 * vertices[v1 * 3 + 0] +
                   b2 * vertices[v2 * 3 + 0] + b3 * vertices[v3 * 3 + 0];
        float py = b0 * vertices[v0 * 3 + 1] + b1 * vertices[v1 * 3 + 1] +
                   b2 * vertices[v2 * 3 + 1] + b3 * vertices[v3 * 3 + 1];
        float pz = b0 * vertices[v0 * 3 + 2] + b1 * vertices[v1 * 3 + 2] +
                   b2 * vertices[v2 * 3 + 2] + b3 * vertices[v3 * 3 + 2];

        // Compute view direction
        float dx = px - cam_pos[0];
        float dy = py - cam_pos[1];
        float dz = pz - cam_pos[2];
        float len = sqrtf(dx * dx + dy * dy + dz * dz);
        float inv_len = (len > 1e-8f) ? (1.0f / len) : 0.0f;

        int out_idx = out_offset + k;
        positions[out_idx * 3 + 0] = px;
        positions[out_idx * 3 + 1] = py;
        positions[out_idx * 3 + 2] = pz;
        directions[out_idx * 3 + 0] = dx * inv_len;
        directions[out_idx * 3 + 1] = dy * inv_len;
        directions[out_idx * 3 + 2] = dz * inv_len;
    }
}

/**
 * @brief Compute prefix sum of intersection counts
 */
__global__ void prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ prefix_sum,
    size_t n) {

    // Simple sequential prefix sum for single block
    // For large arrays, use thrust::inclusive_scan
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += counts[i];
            prefix_sum[i] = sum;
        }
    }
}

/**
 * @brief Propagate image gradients to intersection points
 *
 * For simplified backward: divide output pixel gradient equally among all
 * intersections at that pixel. Stores RGB gradient; alpha gradient is zero.
 */
__global__ void propagate_gradients_kernel(
    const float* __restrict__ grad_rgb_hwc,    // [H, W, 3] output gradient (HWC)
    const int32_t* __restrict__ num_intersects, // [H*W] intersections per pixel
    const int32_t* __restrict__ prefix_sum,    // [H*W] prefix sum
    float* __restrict__ grad_features,         // [total, 4] gradient per sample (RGB + alpha)
    int width,
    int height,
    int max_K) {

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    int num_k = num_intersects[pixel_idx];
    if (num_k == 0) return;

    // Get output gradient for this pixel
    float grad_r = grad_rgb_hwc[pixel_idx * 3 + 0];
    float grad_g = grad_rgb_hwc[pixel_idx * 3 + 1];
    float grad_b = grad_rgb_hwc[pixel_idx * 3 + 2];

    // Divide gradient among all intersections at this pixel
    float scale = 1.0f / static_cast<float>(num_k);

    int out_offset = (pixel_idx > 0) ? prefix_sum[pixel_idx - 1] : 0;

    for (int k = 0; k < num_k; ++k) {
        int out_idx = out_offset + k;
        grad_features[out_idx * 4 + 0] = grad_r * scale;
        grad_features[out_idx * 4 + 1] = grad_g * scale;
        grad_features[out_idx * 4 + 2] = grad_b * scale;
        grad_features[out_idx * 4 + 3] = 0.0f;  // No alpha gradient in simplified version
    }
}

// ------------------------------
// VERTEX GRADIENT KERNEL
// ------------------------------

/**
 * @brief Compute vertex gradients through barycentric interpolation
 *
 * For each intersection, the gradient flows from grad_features (RGB gradient per sample)
 * through barycentric interpolation to the 4 vertices of the tetrahedron.
 *
 * The barycentric interpolation is:
 *   position = sum(bary[i] * vertex[i] for i in 0..3)
 *
 * Therefore:
 *   d_position/d_vertex[i] = bary[i] * I (identity matrix)
 *   d_loss/d_vertex[i] = d_loss/d_position * bary[i]
 *
 * Since grad_features contains RGB gradient per sample, we interpret this as:
 *   - The position gradient (through the feature network) is approximated by grad_features
 *   - Each vertex receives bary[i] * grad_features contribution
 *
 * Gradients are accumulated across all intersections using atomic operations.
 */
__global__ void vertex_gradients_kernel(
    const float* __restrict__ grad_features,      // [total, 4] gradient per sample (RGB + alpha)
    const float* __restrict__ barycentrics,       // [H*W*K, 4] barycentric coords
    const int64_t* __restrict__ tet_indices,      // [H*W*K] tet index per sample
    const int64_t* __restrict__ tetrahedra,       // [T, 4] vertex indices per tet
    const int32_t* __restrict__ num_intersects,   // [H*W] intersections per pixel
    const int32_t* __restrict__ prefix_sum,       // [H*W] prefix sum of intersections
    float* __restrict__ grad_vertices,            // [V, 3] output gradient per vertex
    int width,
    int height,
    int max_K,
    size_t num_vertices) {

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= width * height) return;

    int num_k = num_intersects[pixel_idx];
    if (num_k == 0) return;

    int out_offset = (pixel_idx > 0) ? prefix_sum[pixel_idx - 1] : 0;

    for (int k = 0; k < num_k; ++k) {
        int sample_idx = pixel_idx * max_K + k;
        int grad_idx = out_offset + k;

        int64_t tet_idx = tet_indices[sample_idx];
        if (tet_idx < 0) continue;

        // Get barycentric coordinates for this intersection
        float b0 = barycentrics[sample_idx * 4 + 0];
        float b1 = barycentrics[sample_idx * 4 + 1];
        float b2 = barycentrics[sample_idx * 4 + 2];
        float b3 = barycentrics[sample_idx * 4 + 3];

        // Get gradient for this sample (RGB components form position gradient)
        // The gradient flows through the feature network to position
        float grad_x = grad_features[grad_idx * 4 + 0];
        float grad_y = grad_features[grad_idx * 4 + 1];
        float grad_z = grad_features[grad_idx * 4 + 2];

        // Get vertex indices for this tetrahedron
        int64_t v0 = tetrahedra[tet_idx * 4 + 0];
        int64_t v1 = tetrahedra[tet_idx * 4 + 1];
        int64_t v2 = tetrahedra[tet_idx * 4 + 2];
        int64_t v3 = tetrahedra[tet_idx * 4 + 3];

        // Accumulate gradients to vertices using barycentric weights
        // d_loss/d_vertex[i] = bary[i] * d_loss/d_position
        if (v0 >= 0 && v0 < static_cast<int64_t>(num_vertices)) {
            atomicAdd(&grad_vertices[v0 * 3 + 0], b0 * grad_x);
            atomicAdd(&grad_vertices[v0 * 3 + 1], b0 * grad_y);
            atomicAdd(&grad_vertices[v0 * 3 + 2], b0 * grad_z);
        }

        if (v1 >= 0 && v1 < static_cast<int64_t>(num_vertices)) {
            atomicAdd(&grad_vertices[v1 * 3 + 0], b1 * grad_x);
            atomicAdd(&grad_vertices[v1 * 3 + 1], b1 * grad_y);
            atomicAdd(&grad_vertices[v1 * 3 + 2], b1 * grad_z);
        }

        if (v2 >= 0 && v2 < static_cast<int64_t>(num_vertices)) {
            atomicAdd(&grad_vertices[v2 * 3 + 0], b2 * grad_x);
            atomicAdd(&grad_vertices[v2 * 3 + 1], b2 * grad_y);
            atomicAdd(&grad_vertices[v2 * 3 + 2], b2 * grad_z);
        }

        if (v3 >= 0 && v3 < static_cast<int64_t>(num_vertices)) {
            atomicAdd(&grad_vertices[v3 * 3 + 0], b3 * grad_x);
            atomicAdd(&grad_vertices[v3 * 3 + 1], b3 * grad_y);
            atomicAdd(&grad_vertices[v3 * 3 + 2], b3 * grad_z);
        }
    }
}

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    alpha_blend_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        rgb, alpha, depths, num_intersects, background,
        output_rgb, output_alpha,
        width, height, max_K, alpha_threshold, background_is_image);
}

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
    void* stream) {

    dim3 block_size(16, 16);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    alpha_blend_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        rgb, alpha, depths, num_intersects,
        grad_output_rgb, grad_output_alpha,
        grad_rgb, grad_alpha,
        width, height, max_K, alpha_threshold);
}

void launch_sort_intersections(
    float* depths,
    int64_t* tet_indices,
    float* barycentrics,
    const int32_t* num_intersects,
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

    sort_intersections_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        depths, tet_indices, barycentrics, num_intersects,
        width, height, max_K);
}

void launch_project_vertices(
    const float* vertices,
    const float* R,
    const float* T,
    const float* intrinsics,
    float* projected,
    size_t num_vertices,
    float near_clip,
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_vertices) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    project_vertices_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, R, T, intrinsics, projected, num_vertices, near_clip);
}

void launch_compute_tet_bounds(
    const int64_t* tetrahedra,
    const float* projected,
    float* bounds,
    size_t num_tets,
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_tets) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    compute_tet_bounds_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        tetrahedra, projected, bounds, num_tets);
}

void launch_accumulate_gradient_norms(
    const float* gradients,
    float* accum,
    size_t num_vertices,
    void* stream) {

    int block_size = 256;
    int grid_size = (static_cast<int>(num_vertices) + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    accumulate_gradient_norms_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        gradients, accum, num_vertices);
}

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
    void* stream) {

    int num_pixels = width * height;
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    reconstruct_intersections_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        vertices, tetrahedra, barycentrics, tet_indices, num_intersects,
        cam_pos, positions, directions, prefix_sum, width, height, max_K);
}

void launch_prefix_sum(
    const int32_t* counts,
    int32_t* prefix_sum,
    size_t n,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    prefix_sum_kernel<<<1, 1, 0, cuda_stream>>>(counts, prefix_sum, n);
}

void launch_propagate_gradients(
    const float* grad_rgb_hwc,
    const int32_t* num_intersects,
    const int32_t* prefix_sum,
    float* grad_features,
    int width,
    int height,
    int max_K,
    void* stream) {

    int num_pixels = width * height;
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    propagate_gradients_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        grad_rgb_hwc, num_intersects, prefix_sum, grad_features,
        width, height, max_K);
}

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
    void* stream) {

    int num_pixels = width * height;
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    vertex_gradients_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        grad_features, barycentrics, tet_indices, tetrahedra,
        num_intersects, prefix_sum, grad_vertices,
        width, height, max_K, num_vertices);
}

} // namespace lfs::tetra::cuda
