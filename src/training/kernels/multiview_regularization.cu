/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/multiview_regularization.cuh"
#include <cmath>

namespace lfs::training::kernels {

    // =============================================================================
    // Device helper functions
    // =============================================================================

    /**
     * @brief Bilinear interpolation sampling from 2D image
     */
    __device__ inline float bilinear_sample(
        const float* image,
        float u, float v,
        int height, int width,
        int channel = 0,
        int num_channels = 1) {

        // Clamp coordinates
        u = fmaxf(0.0f, fminf(static_cast<float>(width - 1), u));
        v = fmaxf(0.0f, fminf(static_cast<float>(height - 1), v));

        int u0 = static_cast<int>(floorf(u));
        int v0 = static_cast<int>(floorf(v));
        int u1 = min(u0 + 1, width - 1);
        int v1 = min(v0 + 1, height - 1);

        float du = u - static_cast<float>(u0);
        float dv = v - static_cast<float>(v0);

        // Sample four corners
        float val00 = image[(v0 * width + u0) * num_channels + channel];
        float val01 = image[(v0 * width + u1) * num_channels + channel];
        float val10 = image[(v1 * width + u0) * num_channels + channel];
        float val11 = image[(v1 * width + u1) * num_channels + channel];

        // Bilinear interpolation
        return (1.0f - dv) * ((1.0f - du) * val00 + du * val01) +
               dv * ((1.0f - du) * val10 + du * val11);
    }

    /**
     * @brief Check if coordinates are within image bounds
     */
    __device__ inline bool in_bounds(float u, float v, int width, int height) {
        return u >= 0.0f && u < static_cast<float>(width) &&
               v >= 0.0f && v < static_cast<float>(height);
    }

    /**
     * @brief 3x3 matrix-vector multiplication
     */
    __device__ inline void mat3_vec3(const float* M, const float* v, float* out) {
        out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
        out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
        out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
    }

    // =============================================================================
    // Forward depth warp kernel
    // =============================================================================

    __global__ void depth_warp_forward_kernel(
        const float* __restrict__ ref_depth,
        const float* __restrict__ ref_normal,
        float* __restrict__ warped_u,
        float* __restrict__ warped_v,
        float fx_ref, float fy_ref, float cx_ref, float cy_ref,
        float fx_neighbor, float fy_neighbor, float cx_neighbor, float cy_neighbor,
        const float* __restrict__ R_rel,
        const float* __restrict__ t_rel,
        int height, int width,
        float min_depth) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = height * width;

        if (idx >= total) return;

        int v_coord = idx / width;
        int u_coord = idx % width;

        float depth = ref_depth[idx];

        // Invalid depth: set invalid coordinates
        if (depth < min_depth) {
            warped_u[idx] = -1.0f;
            warped_v[idx] = -1.0f;
            return;
        }

        // Step 1: Unproject to 3D in reference camera coordinates
        // X = depth * K_ref^-1 * [u, v, 1]^T
        float X_ref[3];
        X_ref[0] = depth * (static_cast<float>(u_coord) - cx_ref) / fx_ref;
        X_ref[1] = depth * (static_cast<float>(v_coord) - cy_ref) / fy_ref;
        X_ref[2] = depth;

        // Step 2: Transform to neighbor camera coordinates
        // X_neighbor = R_rel * X_ref + t_rel
        float X_neighbor[3];
        mat3_vec3(R_rel, X_ref, X_neighbor);
        X_neighbor[0] += t_rel[0];
        X_neighbor[1] += t_rel[1];
        X_neighbor[2] += t_rel[2];

        // Step 3: Project to neighbor image
        // [u', v'] = K_neighbor * X_neighbor / z
        if (X_neighbor[2] < min_depth) {
            warped_u[idx] = -1.0f;
            warped_v[idx] = -1.0f;
            return;
        }

        float u_warped = fx_neighbor * X_neighbor[0] / X_neighbor[2] + cx_neighbor;
        float v_warped = fy_neighbor * X_neighbor[1] / X_neighbor[2] + cy_neighbor;

        warped_u[idx] = u_warped;
        warped_v[idx] = v_warped;
    }

    void launch_depth_warp_forward(
        const float* ref_depth,
        const float* ref_normal,
        float* warped_u,
        float* warped_v,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel,
        const float* t_rel,
        int height,
        int width,
        float min_depth,
        cudaStream_t stream) {

        int total = height * width;
        int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        // Copy R_rel and t_rel to device
        float* d_R_rel;
        float* d_t_rel;
        cudaMalloc(&d_R_rel, 9 * sizeof(float));
        cudaMalloc(&d_t_rel, 3 * sizeof(float));
        cudaMemcpyAsync(d_R_rel, R_rel, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_t_rel, t_rel, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        depth_warp_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
            ref_depth, ref_normal,
            warped_u, warped_v,
            ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy,
            neighbor_cam.fx, neighbor_cam.fy, neighbor_cam.cx, neighbor_cam.cy,
            d_R_rel, d_t_rel,
            height, width, min_depth);

        // Free temporary device memory (async-safe since we're done with kernel)
        cudaFree(d_R_rel);
        cudaFree(d_t_rel);
    }

    // =============================================================================
    // Inverse depth warp kernel (for cycle consistency)
    // =============================================================================

    __global__ void depth_warp_inverse_kernel(
        const float* __restrict__ warped_u,
        const float* __restrict__ warped_v,
        const float* __restrict__ neighbor_depth,
        float* __restrict__ cycle_u,
        float* __restrict__ cycle_v,
        float fx_ref, float fy_ref, float cx_ref, float cy_ref,
        float fx_neighbor, float fy_neighbor, float cx_neighbor, float cy_neighbor,
        const float* __restrict__ R_rel_inv,
        const float* __restrict__ t_rel_inv,
        int ref_height, int ref_width,
        int neighbor_height, int neighbor_width,
        float min_depth) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = ref_height * ref_width;

        if (idx >= total) return;

        int v_ref = idx / ref_width;
        int u_ref = idx % ref_width;

        float u_warped = warped_u[idx];
        float v_warped = warped_v[idx];

        // Invalid forward warp: propagate invalid
        if (u_warped < 0.0f || v_warped < 0.0f ||
            !in_bounds(u_warped, v_warped, neighbor_width, neighbor_height)) {
            cycle_u[idx] = static_cast<float>(u_ref);  // Return original coords (zero error)
            cycle_v[idx] = static_cast<float>(v_ref);
            return;
        }

        // Sample depth from neighbor at warped location
        float depth_neighbor = bilinear_sample(neighbor_depth, u_warped, v_warped,
                                                neighbor_height, neighbor_width);

        if (depth_neighbor < min_depth) {
            cycle_u[idx] = static_cast<float>(u_ref);
            cycle_v[idx] = static_cast<float>(v_ref);
            return;
        }

        // Unproject from neighbor to 3D
        float X_neighbor[3];
        X_neighbor[0] = depth_neighbor * (u_warped - cx_neighbor) / fx_neighbor;
        X_neighbor[1] = depth_neighbor * (v_warped - cy_neighbor) / fy_neighbor;
        X_neighbor[2] = depth_neighbor;

        // Transform back to reference coordinates
        float X_ref[3];
        mat3_vec3(R_rel_inv, X_neighbor, X_ref);
        X_ref[0] += t_rel_inv[0];
        X_ref[1] += t_rel_inv[1];
        X_ref[2] += t_rel_inv[2];

        // Project to reference image
        if (X_ref[2] < min_depth) {
            cycle_u[idx] = static_cast<float>(u_ref);
            cycle_v[idx] = static_cast<float>(v_ref);
            return;
        }

        float u_cycle = fx_ref * X_ref[0] / X_ref[2] + cx_ref;
        float v_cycle = fy_ref * X_ref[1] / X_ref[2] + cy_ref;

        cycle_u[idx] = u_cycle;
        cycle_v[idx] = v_cycle;
    }

    void launch_depth_warp_inverse(
        const float* warped_u,
        const float* warped_v,
        const float* neighbor_depth,
        float* cycle_u,
        float* cycle_v,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel_inv,
        const float* t_rel_inv,
        int ref_height,
        int ref_width,
        int neighbor_height,
        int neighbor_width,
        float min_depth,
        cudaStream_t stream) {

        int total = ref_height * ref_width;
        int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        // Copy matrices to device
        float* d_R_rel_inv;
        float* d_t_rel_inv;
        cudaMalloc(&d_R_rel_inv, 9 * sizeof(float));
        cudaMalloc(&d_t_rel_inv, 3 * sizeof(float));
        cudaMemcpyAsync(d_R_rel_inv, R_rel_inv, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_t_rel_inv, t_rel_inv, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        depth_warp_inverse_kernel<<<num_blocks, block_size, 0, stream>>>(
            warped_u, warped_v, neighbor_depth,
            cycle_u, cycle_v,
            ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy,
            neighbor_cam.fx, neighbor_cam.fy, neighbor_cam.cx, neighbor_cam.cy,
            d_R_rel_inv, d_t_rel_inv,
            ref_height, ref_width,
            neighbor_height, neighbor_width,
            min_depth);

        cudaFree(d_R_rel_inv);
        cudaFree(d_t_rel_inv);
    }

    // =============================================================================
    // Cycle error and confidence weights kernel
    // =============================================================================

    __global__ void cycle_error_weights_kernel(
        const float* __restrict__ cycle_u,
        const float* __restrict__ cycle_v,
        float* __restrict__ error_out,
        float* __restrict__ weight_out,
        int height, int width,
        float error_threshold) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = height * width;

        if (idx >= total) return;

        int v_coord = idx / width;
        int u_coord = idx % width;

        float u_cycle = cycle_u[idx];
        float v_cycle = cycle_v[idx];

        // Compute reprojection error
        float du = u_cycle - static_cast<float>(u_coord);
        float dv = v_cycle - static_cast<float>(v_coord);
        float error = sqrtf(du * du + dv * dv);

        error_out[idx] = error;

        // Compute confidence weight: exp(-error) if error < threshold, else 0
        if (error < error_threshold) {
            weight_out[idx] = expf(-error);
        } else {
            weight_out[idx] = 0.0f;
        }
    }

    void launch_cycle_error_and_weights(
        const float* cycle_u,
        const float* cycle_v,
        float* error_out,
        float* weight_out,
        int height,
        int width,
        float error_threshold,
        cudaStream_t stream) {

        int total = height * width;
        int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        cycle_error_weights_kernel<<<num_blocks, block_size, 0, stream>>>(
            cycle_u, cycle_v, error_out, weight_out,
            height, width, error_threshold);
    }

    // =============================================================================
    // NCC matching kernel
    // =============================================================================

    __global__ void ncc_matching_kernel(
        const float* __restrict__ ref_image,
        const float* __restrict__ neighbor_image,
        const float* __restrict__ warped_u,
        const float* __restrict__ warped_v,
        float* __restrict__ ncc_out,
        int ref_height, int ref_width,
        int neighbor_height, int neighbor_width,
        int channels,
        int patch_radius) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = ref_height * ref_width;

        if (idx >= total) return;

        int v_ref = idx / ref_width;
        int u_ref = idx % ref_width;

        float u_warped = warped_u[idx];
        float v_warped = warped_v[idx];

        // Invalid warp or out of bounds for patch
        if (u_warped < 0.0f || v_warped < 0.0f ||
            u_ref - patch_radius < 0 || u_ref + patch_radius >= ref_width ||
            v_ref - patch_radius < 0 || v_ref + patch_radius >= ref_height ||
            u_warped - patch_radius < 0.0f || u_warped + patch_radius >= neighbor_width ||
            v_warped - patch_radius < 0.0f || v_warped + patch_radius >= neighbor_height) {
            ncc_out[idx] = 0.0f;  // No correlation
            return;
        }

        // Compute NCC over the patch for all channels
        float sum_ref = 0.0f;
        float sum_neighbor = 0.0f;
        float sum_ref_sq = 0.0f;
        float sum_neighbor_sq = 0.0f;
        float sum_cross = 0.0f;
        int count = 0;

        for (int dv = -patch_radius; dv <= patch_radius; ++dv) {
            for (int du = -patch_radius; du <= patch_radius; ++du) {
                int u_p = u_ref + du;
                int v_p = v_ref + dv;
                float u_n = u_warped + static_cast<float>(du);
                float v_n = v_warped + static_cast<float>(dv);

                for (int c = 0; c < channels; ++c) {
                    // Reference pixel
                    float val_ref = ref_image[(v_p * ref_width + u_p) * channels + c];

                    // Neighbor pixel (bilinear interpolation)
                    float val_neighbor = bilinear_sample(
                        neighbor_image, u_n, v_n,
                        neighbor_height, neighbor_width, c, channels);

                    sum_ref += val_ref;
                    sum_neighbor += val_neighbor;
                    sum_ref_sq += val_ref * val_ref;
                    sum_neighbor_sq += val_neighbor * val_neighbor;
                    sum_cross += val_ref * val_neighbor;
                    count++;
                }
            }
        }

        // Compute NCC
        float mean_ref = sum_ref / static_cast<float>(count);
        float mean_neighbor = sum_neighbor / static_cast<float>(count);

        float var_ref = sum_ref_sq / static_cast<float>(count) - mean_ref * mean_ref;
        float var_neighbor = sum_neighbor_sq / static_cast<float>(count) - mean_neighbor * mean_neighbor;
        float covar = sum_cross / static_cast<float>(count) - mean_ref * mean_neighbor;

        float denom = sqrtf(fmaxf(var_ref, 1e-8f) * fmaxf(var_neighbor, 1e-8f));
        float ncc = covar / denom;

        // Clamp to [-1, 1]
        ncc_out[idx] = fmaxf(-1.0f, fminf(1.0f, ncc));
    }

    void launch_ncc_matching(
        const float* ref_image,
        const float* neighbor_image,
        const float* warped_u,
        const float* warped_v,
        float* ncc_out,
        int ref_height,
        int ref_width,
        int neighbor_height,
        int neighbor_width,
        int channels,
        int patch_size,
        cudaStream_t stream) {

        int total = ref_height * ref_width;
        int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;
        int patch_radius = patch_size / 2;

        ncc_matching_kernel<<<num_blocks, block_size, 0, stream>>>(
            ref_image, neighbor_image,
            warped_u, warped_v,
            ncc_out,
            ref_height, ref_width,
            neighbor_height, neighbor_width,
            channels, patch_radius);
    }

    // =============================================================================
    // Combined loss reduction kernel
    // =============================================================================

    __global__ void multiview_loss_kernel(
        const float* __restrict__ reprojection_error,
        const float* __restrict__ ncc_scores,
        const float* __restrict__ confidence_weights,
        float* __restrict__ partial_loss,
        float* __restrict__ partial_weight,
        int height, int width,
        float photometric_weight,
        float geometric_weight) {

        float local_loss = 0.0f;
        float local_weight = 0.0f;
        int total = height * width;

        // Grid-stride loop
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += blockDim.x * gridDim.x) {

            float weight = confidence_weights[idx];
            if (weight > 0.0f) {
                float error = reprojection_error[idx];
                float ncc = ncc_scores[idx];

                // Loss = photometric_weight * (1 - NCC) + geometric_weight * error
                float pixel_loss = photometric_weight * (1.0f - ncc) +
                                   geometric_weight * error;

                local_loss += weight * pixel_loss;
                local_weight += weight;
            }
        }

        // Block-level warp reduction
        local_loss = lfs::core::warp_ops::block_reduce_sum(local_loss);
        local_weight = lfs::core::warp_ops::block_reduce_sum(local_weight);

        if (threadIdx.x == 0) {
            partial_loss[blockIdx.x] = local_loss;
            partial_weight[blockIdx.x] = local_weight;
        }
    }

    __global__ void final_loss_reduce_kernel(
        const float* __restrict__ partial_loss,
        const float* __restrict__ partial_weight,
        float* __restrict__ loss_out,
        int num_blocks) {

        float sum_loss = 0.0f;
        float sum_weight = 0.0f;

        // Grid-stride loop to handle all partial sums
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum_loss += partial_loss[i];
            sum_weight += partial_weight[i];
        }

        // Block-level warp reduction
        sum_loss = lfs::core::warp_ops::block_reduce_sum(sum_loss);
        sum_weight = lfs::core::warp_ops::block_reduce_sum(sum_weight);

        if (threadIdx.x == 0) {
            // Normalize by total weight
            if (sum_weight > 1e-6f) {
                loss_out[0] = sum_loss / sum_weight;
            } else {
                loss_out[0] = 0.0f;
            }
        }
    }

    void launch_multiview_loss_reduction(
        const float* reprojection_error,
        const float* ncc_scores,
        const float* confidence_weights,
        float* loss_out,
        float* partial_sums,
        int height,
        int width,
        float photometric_weight,
        float geometric_weight,
        cudaStream_t stream) {

        int total = height * width;
        int block_size = 256;
        int num_blocks = std::min((total + block_size - 1) / block_size, 1024);

        // partial_sums layout: [0..num_blocks-1] = loss, [num_blocks..2*num_blocks-1] = weight
        float* partial_loss = partial_sums;
        float* partial_weight = partial_sums + num_blocks;

        // First pass: compute partial sums
        multiview_loss_kernel<<<num_blocks, block_size, 0, stream>>>(
            reprojection_error, ncc_scores, confidence_weights,
            partial_loss, partial_weight,
            height, width,
            photometric_weight, geometric_weight);

        // Second pass: reduce partial sums
        final_loss_reduce_kernel<<<1, block_size, 0, stream>>>(
            partial_loss, partial_weight, loss_out, num_blocks);
    }

    // =============================================================================
    // Backward kernel (gradient computation)
    // =============================================================================

    __global__ void multiview_backward_kernel(
        const float* __restrict__ reprojection_error,
        const float* __restrict__ cycle_u,
        const float* __restrict__ cycle_v,
        const float* __restrict__ confidence_weights,
        const float* __restrict__ ref_depth,
        const float* __restrict__ ref_normal,
        float* __restrict__ depth_grad,
        float* __restrict__ normal_grad,
        float fx_ref, float fy_ref, float cx_ref, float cy_ref,
        float fx_neighbor, float fy_neighbor, float cx_neighbor, float cy_neighbor,
        const float* __restrict__ R_rel,
        const float* __restrict__ t_rel,
        int height, int width,
        float geometric_weight,
        float total_weight) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = height * width;

        if (idx >= total) return;

        int v_coord = idx / width;
        int u_coord = idx % width;

        float weight = confidence_weights[idx];
        if (weight < 1e-6f) return;  // No contribution

        float error = reprojection_error[idx];
        float depth = ref_depth[idx];

        if (depth < 0.01f) return;  // Invalid depth

        // Gradient scale factor
        float grad_scale = geometric_weight * weight / total_weight;

        // Compute gradient of error w.r.t. cycle coordinates
        float u_cyc = cycle_u[idx];
        float v_cyc = cycle_v[idx];
        float du = u_cyc - static_cast<float>(u_coord);
        float dv = v_cyc - static_cast<float>(v_coord);

        // d(error)/d(cycle_u) = du / error, d(error)/d(cycle_v) = dv / error
        // (for error = sqrt(du^2 + dv^2))
        float inv_error = (error > 1e-6f) ? (1.0f / error) : 0.0f;
        float d_cycle_u = du * inv_error;
        float d_cycle_v = dv * inv_error;

        // Chain rule through projection and transformation
        // This is a simplified gradient that captures the main effect:
        // d(error)/d(depth) approx = (d_cycle_u * fx + d_cycle_v * fy) / depth^2

        float depth_contribution = (d_cycle_u * fx_ref + d_cycle_v * fy_ref) / (depth * depth);

        // Accumulate gradient
        atomicAdd(&depth_grad[idx], grad_scale * depth_contribution);

        // Normal gradient is more complex - simplified here
        // The full gradient involves the homography Jacobian
        // For now, we use a simple approximation based on depth gradient
        float normal_scale = depth_contribution * 0.1f;  // Scaling factor

        atomicAdd(&normal_grad[idx * 3 + 0], grad_scale * normal_scale * R_rel[0]);
        atomicAdd(&normal_grad[idx * 3 + 1], grad_scale * normal_scale * R_rel[4]);
        atomicAdd(&normal_grad[idx * 3 + 2], grad_scale * normal_scale * R_rel[8]);
    }

    void launch_multiview_backward(
        const float* reprojection_error,
        const float* cycle_u,
        const float* cycle_v,
        const float* confidence_weights,
        const float* ref_depth,
        const float* ref_normal,
        float* depth_grad,
        float* normal_grad,
        const CameraParams& ref_cam,
        const CameraParams& neighbor_cam,
        const float* R_rel,
        const float* t_rel,
        int height,
        int width,
        float geometric_weight,
        float total_weight,
        cudaStream_t stream) {

        int total = height * width;
        int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        // Copy R_rel and t_rel to device
        float* d_R_rel;
        float* d_t_rel;
        cudaMalloc(&d_R_rel, 9 * sizeof(float));
        cudaMalloc(&d_t_rel, 3 * sizeof(float));
        cudaMemcpyAsync(d_R_rel, R_rel, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_t_rel, t_rel, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        multiview_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
            reprojection_error, cycle_u, cycle_v,
            confidence_weights, ref_depth, ref_normal,
            depth_grad, normal_grad,
            ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy,
            neighbor_cam.fx, neighbor_cam.fy, neighbor_cam.cx, neighbor_cam.cy,
            d_R_rel, d_t_rel,
            height, width,
            geometric_weight, total_weight);

        cudaFree(d_R_rel);
        cudaFree(d_t_rel);
    }

} // namespace lfs::training::kernels
