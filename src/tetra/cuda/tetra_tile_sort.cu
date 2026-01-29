/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tetra_tile_sort.cu
 * @brief CUB-based sorting utilities for tile-based tetra rasterization
 *
 * Uses CUB library for high-performance GPU radix sort and prefix sum.
 */

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

namespace lfs::tetra::cuda {

// Wrapper macro for CUB temporary storage allocation
#define CUB_WRAPPER_TETRA(func, ...)                                           \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                        \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                       \
        void* d_temp_storage = nullptr;                                       \
        cudaMalloc(&d_temp_storage, temp_storage_bytes);                      \
        func(d_temp_storage, temp_storage_bytes, __VA_ARGS__);                \
        cudaFree(d_temp_storage);                                             \
    } while (0)

void radix_sort_tet_tiles(
    int64_t n_pairs,
    int64_t* keys_in,
    int32_t* values_in,
    int64_t* keys_out,
    int32_t* values_out,
    int num_key_bits,
    void* stream) {

    if (n_pairs <= 0) {
        return;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    cub::DoubleBuffer<int64_t> d_keys(keys_in, keys_out);
    cub::DoubleBuffer<int32_t> d_values(values_in, values_out);

    // Get temporary storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        d_keys, d_values, n_pairs,
        0, num_key_bits, cuda_stream);

    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values, n_pairs,
        0, num_key_bits, cuda_stream);

    cudaFree(d_temp_storage);

    // Copy results to output buffers if needed (CUB may have swapped them)
    if (d_keys.selector == 0) {
        cudaMemcpyAsync(keys_out, keys_in,
                        n_pairs * sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_stream);
    }
    if (d_values.selector == 0) {
        cudaMemcpyAsync(values_out, values_in,
                        n_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice, cuda_stream);
    }
}

void compute_prefix_sum_int32_to_int64(
    const int32_t* input,
    int64_t* output,
    size_t n_elements,
    void* stream) {

    if (n_elements == 0) {
        return;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    // Cast int32 to int64 during scan
    auto cast_op = [] __host__ __device__(int32_t x) { return static_cast<int64_t>(x); };
    auto cast_iter = thrust::make_transform_iterator(input, cast_op);

    // Get temporary storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cast_iter, output, n_elements, cuda_stream);

    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Compute prefix sum
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        cast_iter, output, n_elements, cuda_stream);

    cudaFree(d_temp_storage);
}

int64_t get_last_prefix_sum_value(
    const int64_t* prefix_sum,
    size_t n_elements,
    void* stream) {

    if (n_elements == 0) {
        return 0;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    int64_t result;
    cudaMemcpyAsync(&result, prefix_sum + n_elements - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    return result;
}

} // namespace lfs::tetra::cuda
