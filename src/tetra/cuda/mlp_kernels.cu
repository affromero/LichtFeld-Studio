/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file mlp_kernels.cu
 * @brief CUDA kernels for MLP forward pass with tiled matrix multiplication
 *
 * Implements batched matrix multiplication with ReLU and sigmoid activations
 * for neural network inference on GPU.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace lfs::tetra::cuda {

namespace {

// Tile size for shared memory tiling
constexpr int TILE_SIZE = 16;

/**
 * @brief Tiled matrix multiplication with ReLU activation
 *
 * Computes: output = ReLU(input @ weight^T + bias)
 * where input is [N, in_features] and weight is [out_features, in_features]
 *
 * Uses shared memory tiling for improved memory access patterns.
 */
__global__ void matmul_relu_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [out_features, in_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    // Shared memory for tiles
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    // Output element position
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Sample index in batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Output feature index

    float sum = 0.0f;

    // Number of tiles needed to cover in_features dimension
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile: input[row, t*TILE_SIZE + threadIdx.x]
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[col, t*TILE_SIZE + threadIdx.y]
        // Note: We want weight^T, so we load weight[col, k] into tile_weight[threadIdx.y][threadIdx.x]
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (col < out_features && weight_row < in_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + weight_row];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write output with bias and ReLU activation
    if (row < N && col < out_features) {
        float result = sum + bias[col];
        output[row * out_features + col] = fmaxf(result, 0.0f);
    }
}

/**
 * @brief Tiled matrix multiplication with sigmoid activation
 *
 * Computes: output = sigmoid(input @ weight^T + bias)
 * where input is [N, in_features] and weight is [out_features, in_features]
 *
 * Uses shared memory tiling for improved memory access patterns.
 */
__global__ void matmul_sigmoid_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [out_features, in_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    // Shared memory for tiles
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    // Output element position
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Sample index in batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Output feature index

    float sum = 0.0f;

    // Number of tiles needed to cover in_features dimension
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile: input[row, t*TILE_SIZE + threadIdx.x]
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[col, t*TILE_SIZE + threadIdx.y]
        // Note: We want weight^T, so we load weight[col, k] into tile_weight[threadIdx.y][threadIdx.x]
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (col < out_features && weight_row < in_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * in_features + weight_row];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write output with bias and sigmoid activation
    if (row < N && col < out_features) {
        float result = sum + bias[col];
        // Sigmoid: 1 / (1 + exp(-x))
        // Use numerically stable sigmoid
        if (result >= 0.0f) {
            output[row * out_features + col] = 1.0f / (1.0f + expf(-result));
        } else {
            float exp_x = expf(result);
            output[row * out_features + col] = exp_x / (1.0f + exp_x);
        }
    }
}

/**
 * @brief Simple (non-tiled) matrix multiplication with ReLU for small matrices
 *
 * More efficient for small batch sizes or feature dimensions that don't benefit
 * from tiling overhead.
 */
__global__ void matmul_relu_simple_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [out_features, in_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

    if (row >= N || col >= out_features) return;

    float sum = 0.0f;

    // Compute dot product: input[row, :] @ weight[col, :]^T
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Add bias and apply ReLU
    float result = sum + bias[col];
    output[row * out_features + col] = fmaxf(result, 0.0f);
}

/**
 * @brief Simple (non-tiled) matrix multiplication with sigmoid for small matrices
 */
__global__ void matmul_sigmoid_simple_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [out_features, in_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

    if (row >= N || col >= out_features) return;

    float sum = 0.0f;

    // Compute dot product: input[row, :] @ weight[col, :]^T
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Add bias and apply sigmoid
    float result = sum + bias[col];

    // Numerically stable sigmoid
    if (result >= 0.0f) {
        output[row * out_features + col] = 1.0f / (1.0f + expf(-result));
    } else {
        float exp_x = expf(result);
        output[row * out_features + col] = exp_x / (1.0f + exp_x);
    }
}

/**
 * @brief Determine whether to use tiled or simple kernel
 *
 * Tiling benefits large matrices; for small ones the overhead isn't worth it.
 */
__host__ bool should_use_tiling(int N, int in_features, int out_features) {
    // Use tiling if any dimension is larger than 2x tile size
    constexpr int TILING_THRESHOLD = 2 * TILE_SIZE;
    return (N >= TILING_THRESHOLD) ||
           (in_features >= TILING_THRESHOLD) ||
           (out_features >= TILING_THRESHOLD);
}

} // namespace

// ------------------------------
// PUBLIC API
// ------------------------------

void launch_matmul_relu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, in_features, out_features)) {
        // Tiled kernel
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        matmul_relu_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    } else {
        // Simple kernel for small matrices
        dim3 block_size(16, 16);
        dim3 grid_size(
            (out_features + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );

        matmul_relu_simple_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    }
}

void launch_matmul_sigmoid(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, in_features, out_features)) {
        // Tiled kernel
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        matmul_sigmoid_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    } else {
        // Simple kernel for small matrices
        dim3 block_size(16, 16);
        dim3 grid_size(
            (out_features + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );

        matmul_sigmoid_simple_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    }
}

// ------------------------------
// KERNELS WITH [in_features, out_features] WEIGHT LAYOUT
// (Standard row-major layout matching initialization)
// ------------------------------

namespace {

/**
 * @brief Tiled matrix multiplication with ReLU activation
 *
 * Computes: output = ReLU(input @ weight + bias)
 * where input is [N, in_features] and weight is [in_features, out_features]
 *
 * Uses shared memory tiling for improved memory access patterns.
 */
__global__ void matmul_relu_rowmajor_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [in_features, out_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    // Shared memory for tiles
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    // Output element position
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Sample index in batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Output feature index

    float sum = 0.0f;

    // Number of tiles needed to cover in_features dimension
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile: input[row, t*TILE_SIZE + threadIdx.x]
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[t*TILE_SIZE + threadIdx.y, col]
        // Weight layout is [in_features, out_features]
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (weight_row < in_features && col < out_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[weight_row * out_features + col];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write output with bias and ReLU activation
    if (row < N && col < out_features) {
        float result = sum + bias[col];
        output[row * out_features + col] = fmaxf(result, 0.0f);
    }
}

/**
 * @brief Tiled matrix multiplication without activation
 *
 * Computes: output = input @ weight + bias
 * where input is [N, in_features] and weight is [in_features, out_features]
 */
__global__ void matmul_bias_rowmajor_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [in_features, out_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    // Shared memory for tiles
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    // Output element position
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Sample index in batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Output feature index

    float sum = 0.0f;

    // Number of tiles needed to cover in_features dimension
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load input tile
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[t*TILE_SIZE + threadIdx.y, col]
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (weight_row < in_features && col < out_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[weight_row * out_features + col];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write output with bias (no activation)
    if (row < N && col < out_features) {
        output[row * out_features + col] = sum + bias[col];
    }
}

/**
 * @brief Simple (non-tiled) matrix multiplication with ReLU for small matrices
 */
__global__ void matmul_relu_simple_rowmajor_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [in_features, out_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

    if (row >= N || col >= out_features) return;

    float sum = 0.0f;

    // Compute dot product: input[row, :] @ weight[:, col]
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[k * out_features + col];
    }

    // Add bias and apply ReLU
    float result = sum + bias[col];
    output[row * out_features + col] = fmaxf(result, 0.0f);
}

/**
 * @brief Simple (non-tiled) matrix multiplication with bias only
 */
__global__ void matmul_bias_simple_rowmajor_kernel(
    const float* __restrict__ input,     // [N, in_features]
    const float* __restrict__ weight,    // [in_features, out_features]
    const float* __restrict__ bias,      // [out_features]
    float* __restrict__ output,          // [N, out_features]
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

    if (row >= N || col >= out_features) return;

    float sum = 0.0f;

    // Compute dot product: input[row, :] @ weight[:, col]
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[k * out_features + col];
    }

    // Add bias (no activation)
    output[row * out_features + col] = sum + bias[col];
}

/**
 * @brief Matrix multiplication for backward pass (no transpose on weight)
 *
 * Computes: output = input @ weight^T (where weight is stored as [in_out, out_in])
 * For MLP backward: grad_input = grad_output @ weight^T
 * where weight is [in_features, out_features] in row-major
 *
 * grad_input[n, i] = sum_o grad_output[n, o] * weight[i, o]
 */
__global__ void matmul_backward_kernel(
    const float* __restrict__ grad_output,  // [N, out_features]
    const float* __restrict__ weight,       // [in_features, out_features]
    float* __restrict__ grad_input,         // [N, in_features]
    int N,
    int out_features,
    int in_features) {

    // Shared memory for tiles
    __shared__ float tile_grad[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Sample index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Input feature index

    float sum = 0.0f;

    int num_tiles = (out_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load grad_output tile: grad_output[row, t*TILE_SIZE + threadIdx.x]
        int grad_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && grad_col < out_features) {
            tile_grad[threadIdx.y][threadIdx.x] = grad_output[row * out_features + grad_col];
        } else {
            tile_grad[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile: weight[col, t*TILE_SIZE + threadIdx.y]
        // We want weight^T[t*TILE_SIZE + threadIdx.y, col] = weight[col, t*TILE_SIZE + threadIdx.y]
        int weight_col = t * TILE_SIZE + threadIdx.y;
        if (col < in_features && weight_col < out_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[col * out_features + weight_col];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_grad[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < in_features) {
        grad_input[row * in_features + col] = sum;
    }
}

/**
 * @brief Simple (non-tiled) backward matmul
 */
__global__ void matmul_backward_simple_kernel(
    const float* __restrict__ grad_output,  // [N, out_features]
    const float* __restrict__ weight,       // [in_features, out_features]
    float* __restrict__ grad_input,         // [N, in_features]
    int N,
    int out_features,
    int in_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= in_features) return;

    float sum = 0.0f;
    for (int o = 0; o < out_features; ++o) {
        sum += grad_output[row * out_features + o] * weight[col * out_features + o];
    }
    grad_input[row * in_features + col] = sum;
}

/**
 * @brief Extract first 3 channels and apply sigmoid
 *
 * For MLP output with output_dim > 3, extracts RGB channels and applies sigmoid.
 * output[i, c] = sigmoid(input[i, c]) for c in [0, 1, 2]
 */
__global__ void extract_rgb_sigmoid_kernel(
    const float* __restrict__ input,   // [N, output_dim]
    float* __restrict__ output,        // [N, 3]
    int N, int output_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 3;

    if (idx < total) {
        int n = idx / 3;
        int c = idx % 3;
        float val = input[n * output_dim + c];

        // Numerically stable sigmoid
        if (val >= 0.0f) {
            output[idx] = 1.0f / (1.0f + expf(-val));
        } else {
            float exp_x = expf(val);
            output[idx] = exp_x / (1.0f + exp_x);
        }
    }
}

} // namespace

void launch_matmul_relu_rowmajor(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, in_features, out_features)) {
        // Tiled kernel
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        matmul_relu_rowmajor_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    } else {
        // Simple kernel for small matrices
        dim3 block_size(16, 16);
        dim3 grid_size(
            (out_features + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );

        matmul_relu_simple_rowmajor_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    }
}

void launch_matmul_bias_rowmajor(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, in_features, out_features)) {
        // Tiled kernel
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        matmul_bias_rowmajor_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    } else {
        // Simple kernel for small matrices
        dim3 block_size(16, 16);
        dim3 grid_size(
            (out_features + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );

        matmul_bias_simple_rowmajor_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, weight, bias, output, N, in_features, out_features);
    }
}

void launch_extract_rgb_sigmoid(
    const float* input,
    float* output,
    size_t N,
    int output_dim,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    int total = static_cast<int>(N) * 3;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    extract_rgb_sigmoid_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        input, output, static_cast<int>(N), output_dim);
}

void launch_matmul_backward(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int N,
    int out_features,
    int in_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, out_features, in_features)) {
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (in_features + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE
        );

        matmul_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            grad_output, weight, grad_input, N, out_features, in_features);
    } else {
        dim3 block_size(16, 16);
        dim3 grid_size(
            (in_features + block_size.x - 1) / block_size.x,
            (N + block_size.y - 1) / block_size.y
        );

        matmul_backward_simple_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            grad_output, weight, grad_input, N, out_features, in_features);
    }
}

void launch_mlp_forward(
    const float* input,
    const float* const* weights,
    const float* const* biases,
    float* output,
    float* intermediate,
    size_t N,
    int input_dim,
    int hidden_dim,
    int output_dim,
    int num_hidden_layers,
    bool use_relu,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    // Total layers = num_hidden_layers + 1 (output layer)
    // Layer 0: input_dim -> hidden_dim
    // Layers 1 to num_hidden_layers-1: hidden_dim -> hidden_dim
    // Layer num_hidden_layers: hidden_dim -> output_dim

    const float* layer_input = input;
    float* layer_output = intermediate;
    int num_layers = num_hidden_layers + 1;

    for (int layer = 0; layer < num_layers; ++layer) {
        int in_dim = (layer == 0) ? input_dim : hidden_dim;
        int out_dim = (layer == num_layers - 1) ? output_dim : hidden_dim;
        bool apply_relu_this_layer = use_relu && (layer < num_layers - 1);

        // Set output buffer
        if (layer == num_layers - 1) {
            layer_output = output;
        } else {
            // Intermediate buffer: ping-pong between two halves
            layer_output = intermediate + (layer % 2) * N * hidden_dim;
        }

        // Choose kernel based on activation
        if (apply_relu_this_layer) {
            launch_matmul_relu_rowmajor(
                layer_input, weights[layer], biases[layer], layer_output,
                static_cast<int>(N), in_dim, out_dim, cuda_stream);
        } else {
            launch_matmul_bias_rowmajor(
                layer_input, weights[layer], biases[layer], layer_output,
                static_cast<int>(N), in_dim, out_dim, cuda_stream);
        }

        // Next layer's input is this layer's output
        layer_input = layer_output;
    }
}

void launch_mlp_backward(
    const float* grad_output,
    const float* input,
    const float* intermediate,
    const float* const* weights,
    float* grad_input,
    float** grad_weights,
    float** grad_biases,
    size_t N,
    int input_dim,
    int hidden_dim,
    int output_dim,
    int num_hidden_layers,
    bool use_relu,
    void* stream) {

    // Backward pass placeholder - full implementation needed for training
    (void)grad_output;
    (void)input;
    (void)intermediate;
    (void)weights;
    (void)grad_input;
    (void)grad_weights;
    (void)grad_biases;
    (void)N;
    (void)input_dim;
    (void)hidden_dim;
    (void)output_dim;
    (void)num_hidden_layers;
    (void)use_relu;
    (void)stream;
}

// ------------------------------
// FORWARD WITH CACHING FOR BACKWARD
// ------------------------------

namespace {

/**
 * @brief Tiled matrix multiplication with ReLU, storing pre-activation
 *
 * Computes: output = ReLU(input @ weight + bias)
 * Also stores the pre-activation (input @ weight + bias) for backward pass
 */
__global__ void matmul_relu_cache_kernel(
    const float* __restrict__ input,      // [N, in_features]
    const float* __restrict__ weight,     // [in_features, out_features]
    const float* __restrict__ bias,       // [out_features]
    float* __restrict__ output,           // [N, out_features] - post-activation (ReLU)
    float* __restrict__ pre_activation,   // [N, out_features] - before ReLU
    int N,
    int in_features,
    int out_features) {

    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < N && input_col < in_features) {
            tile_input[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (weight_row < in_features && col < out_features) {
            tile_weight[threadIdx.y][threadIdx.x] = weight[weight_row * out_features + col];
        } else {
            tile_weight[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input[threadIdx.y][k] * tile_weight[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < out_features) {
        float z = sum + bias[col];
        pre_activation[row * out_features + col] = z;
        output[row * out_features + col] = fmaxf(z, 0.0f);
    }
}

/**
 * @brief Simple (non-tiled) matrix multiplication with ReLU and caching
 */
__global__ void matmul_relu_cache_simple_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ pre_activation,
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[k * out_features + col];
    }

    float z = sum + bias[col];
    pre_activation[row * out_features + col] = z;
    output[row * out_features + col] = fmaxf(z, 0.0f);
}

/**
 * @brief ReLU backward kernel: grad_input = grad_output * (pre_activation > 0)
 */
__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ pre_activation,
    float* __restrict__ grad_input,
    size_t N) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] = (pre_activation[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

/**
 * @brief Compute weight gradient: grad_W = input^T @ grad_output
 *
 * input: [N, in_features]
 * grad_output: [N, out_features]
 * grad_weight: [in_features, out_features]
 */
__global__ void weight_grad_kernel(
    const float* __restrict__ input,       // [N, in_features]
    const float* __restrict__ grad_output, // [N, out_features]
    float* __restrict__ grad_weight,       // [in_features, out_features]
    int N,
    int in_features,
    int out_features) {

    __shared__ float tile_input_t[TILE_SIZE][TILE_SIZE];  // Transposed
    __shared__ float tile_grad[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // in_features index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // out_features index

    float sum = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load input^T tile: input[t*TILE_SIZE + threadIdx.x, row]
        // We want input^T[row, t*TILE_SIZE + threadIdx.x] = input[t*TILE_SIZE + threadIdx.x, row]
        int n_idx = t * TILE_SIZE + threadIdx.x;
        if (n_idx < N && row < in_features) {
            tile_input_t[threadIdx.y][threadIdx.x] = input[n_idx * in_features + row];
        } else {
            tile_input_t[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load grad_output tile: grad_output[t*TILE_SIZE + threadIdx.y, col]
        int grad_n = t * TILE_SIZE + threadIdx.y;
        if (grad_n < N && col < out_features) {
            tile_grad[threadIdx.y][threadIdx.x] = grad_output[grad_n * out_features + col];
        } else {
            tile_grad[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_input_t[threadIdx.y][k] * tile_grad[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < in_features && col < out_features) {
        grad_weight[row * out_features + col] = sum;
    }
}

/**
 * @brief Simple weight gradient kernel
 */
__global__ void weight_grad_simple_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weight,
    int N,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // in_features
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // out_features

    if (row >= in_features || col >= out_features) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += input[n * in_features + row] * grad_output[n * out_features + col];
    }
    grad_weight[row * out_features + col] = sum;
}

/**
 * @brief Compute bias gradient: grad_b = sum(grad_output, axis=0)
 */
__global__ void bias_grad_kernel(
    const float* __restrict__ grad_output,  // [N, out_features]
    float* __restrict__ grad_bias,          // [out_features]
    int N,
    int out_features) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_features) return;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += grad_output[n * out_features + col];
    }
    grad_bias[col] = sum;
}

/**
 * @brief Parallel reduction for bias gradient using shared memory
 */
__global__ void bias_grad_parallel_kernel(
    const float* __restrict__ grad_output,  // [N, out_features]
    float* __restrict__ grad_bias,          // [out_features]
    int N,
    int out_features) {

    extern __shared__ float sdata[];

    int col = blockIdx.x;
    int tid = threadIdx.x;
    int grid_size = blockDim.x;

    float sum = 0.0f;
    for (int n = tid; n < N; n += grid_size) {
        sum += grad_output[n * out_features + col];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[col] = sdata[0];
    }
}

} // namespace

void launch_mlp_forward_with_cache(
    const float* input,
    const float* const* weights,
    const float* const* biases,
    float* output,
    float** layer_inputs,
    float** pre_activations,
    size_t N,
    int input_dim,
    int hidden_dim,
    int output_dim,
    int num_hidden_layers,
    bool use_relu,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    int num_layers = num_hidden_layers + 1;
    const float* layer_input = input;

    for (int layer = 0; layer < num_layers; ++layer) {
        int in_dim = (layer == 0) ? input_dim : hidden_dim;
        int out_dim = (layer == num_layers - 1) ? output_dim : hidden_dim;
        bool apply_relu = use_relu && (layer < num_layers - 1);

        float* layer_output = (layer == num_layers - 1) ? output : layer_inputs[layer + 1];
        float* pre_act = pre_activations[layer];

        if (apply_relu) {
            if (should_use_tiling(static_cast<int>(N), in_dim, out_dim)) {
                dim3 block_size(TILE_SIZE, TILE_SIZE);
                dim3 grid_size(
                    (out_dim + TILE_SIZE - 1) / TILE_SIZE,
                    (static_cast<int>(N) + TILE_SIZE - 1) / TILE_SIZE
                );
                matmul_relu_cache_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
                    layer_input, weights[layer], biases[layer],
                    layer_output, pre_act,
                    static_cast<int>(N), in_dim, out_dim);
            } else {
                dim3 block_size(16, 16);
                dim3 grid_size(
                    (out_dim + block_size.x - 1) / block_size.x,
                    (static_cast<int>(N) + block_size.y - 1) / block_size.y
                );
                matmul_relu_cache_simple_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
                    layer_input, weights[layer], biases[layer],
                    layer_output, pre_act,
                    static_cast<int>(N), in_dim, out_dim);
            }
        } else {
            // No activation (output layer) - just linear + bias
            // Store pre-activation (which equals output for linear)
            launch_matmul_bias_rowmajor(
                layer_input, weights[layer], biases[layer], layer_output,
                static_cast<int>(N), in_dim, out_dim, cuda_stream);

            // Copy output to pre_activation (they're the same for linear layer)
            cudaMemcpyAsync(pre_act, layer_output,
                           N * static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToDevice, cuda_stream);
        }

        layer_input = layer_output;
    }
}

void launch_relu_backward(
    const float* grad_output,
    const float* pre_activation,
    float* grad_input,
    size_t N,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    int block_size = 256;
    int grid_size = (static_cast<int>(N) + block_size - 1) / block_size;

    relu_backward_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        grad_output, pre_activation, grad_input, N);
}

void launch_weight_grad(
    const float* input,
    const float* grad_output,
    float* grad_weight,
    int N,
    int in_features,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (should_use_tiling(N, in_features, out_features)) {
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size(
            (out_features + TILE_SIZE - 1) / TILE_SIZE,
            (in_features + TILE_SIZE - 1) / TILE_SIZE
        );
        weight_grad_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, grad_output, grad_weight, N, in_features, out_features);
    } else {
        dim3 block_size(16, 16);
        dim3 grid_size(
            (out_features + block_size.x - 1) / block_size.x,
            (in_features + block_size.y - 1) / block_size.y
        );
        weight_grad_simple_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            input, grad_output, grad_weight, N, in_features, out_features);
    }
}

void launch_bias_grad(
    const float* grad_output,
    float* grad_bias,
    int N,
    int out_features,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (N > 256) {
        // Use parallel reduction for large batches
        int block_size = 256;
        size_t shared_mem = static_cast<size_t>(block_size) * sizeof(float);
        bias_grad_parallel_kernel<<<out_features, block_size, shared_mem, cuda_stream>>>(
            grad_output, grad_bias, N, out_features);
    } else {
        // Simple kernel for small batches
        int block_size = 256;
        int grid_size = (out_features + block_size - 1) / block_size;
        bias_grad_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            grad_output, grad_bias, N, out_features);
    }
}

void launch_mlp_backward_full(
    const float* grad_output,
    const float* const* layer_inputs,
    const float* const* pre_activations,
    const float* const* weights,
    float* grad_input,
    float** grad_weights,
    float** grad_biases,
    size_t N,
    int input_dim,
    int hidden_dim,
    int output_dim,
    int num_hidden_layers,
    bool use_relu,
    void* stream) {

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    int num_layers = num_hidden_layers + 1;

    // Allocate temporary buffer for gradient propagation
    float* grad_z = nullptr;
    float* grad_h = nullptr;
    size_t max_dim = static_cast<size_t>(
        (hidden_dim > output_dim) ? hidden_dim : output_dim);
    cudaMalloc(&grad_z, N * max_dim * sizeof(float));
    cudaMalloc(&grad_h, N * max_dim * sizeof(float));

    // Initialize grad_z with grad_output (last layer has no activation)
    cudaMemcpyAsync(grad_z, grad_output,
                   N * static_cast<size_t>(output_dim) * sizeof(float),
                   cudaMemcpyDeviceToDevice, cuda_stream);

    // Backward through layers in reverse order
    for (int l = num_layers - 1; l >= 0; --l) {
        int in_dim = (l == 0) ? input_dim : hidden_dim;
        int out_dim = (l == num_layers - 1) ? output_dim : hidden_dim;

        // Compute grad_bias
        launch_bias_grad(grad_z, grad_biases[l], static_cast<int>(N), out_dim, cuda_stream);

        // Compute grad_weight: grad_W = layer_inputs[l]^T @ grad_z
        launch_weight_grad(layer_inputs[l], grad_z, grad_weights[l],
                          static_cast<int>(N), in_dim, out_dim, cuda_stream);

        if (l > 0) {
            // Compute grad_h = grad_z @ W^T
            launch_matmul_backward(grad_z, weights[l], grad_h,
                                  static_cast<int>(N), out_dim, in_dim, cuda_stream);

            // Apply ReLU derivative: grad_z_{l-1} = grad_h * (pre_activations[l-1] > 0)
            if (use_relu) {
                launch_relu_backward(grad_h, pre_activations[l - 1], grad_z,
                                    N * static_cast<size_t>(in_dim), cuda_stream);
            } else {
                cudaMemcpyAsync(grad_z, grad_h,
                               N * static_cast<size_t>(in_dim) * sizeof(float),
                               cudaMemcpyDeviceToDevice, cuda_stream);
            }
        } else {
            // First layer: output grad_input
            launch_matmul_backward(grad_z, weights[0], grad_input,
                                  static_cast<int>(N), out_dim, in_dim, cuda_stream);
        }
    }

    cudaFree(grad_z);
    cudaFree(grad_h);
}

} // namespace lfs::tetra::cuda
