/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "optimizer_config.h"
#include "utils.h"

// Forward declare the kernels (defined in adam_api.cu)
namespace fast_lfs::optimizer::kernels::adam {
    __global__ void adam_step_cu(
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const float* param_grad,
        const int n_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp);
}

void fast_lfs::optimizer::adam_step(
    float* param,
    float* exp_avg,
    float* exp_avg_sq,
    const float* param_grad,
    const int n_elements,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {

    // IMPORTANT: Use the SAME kernel as legacy (adam_step_cu), NOT the vectorized version!
    // The vectorized kernel (adam_step_vectorized_cu) has different floating-point rounding
    // behavior which causes divergence from legacy implementation.

    // Check for any prior CUDA errors before kernel launch
    cudaError_t prior_err = cudaDeviceSynchronize();
    if (prior_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error before adam_step_cu: ") + cudaGetErrorString(prior_err));
    }

    const int grid_size = div_round_up(n_elements, config::block_size_adam_step);
    const int block_size = config::block_size_adam_step;

    // Debug: print launch configuration
    printf("[adam_step] Launching kernel: grid=%d, block=%d, n_elements=%d\n", grid_size, block_size, n_elements);
    fflush(stdout);

    // Check if kernel function pointer is valid
    void* kernel_func = (void*)kernels::adam::adam_step_cu;
    if (!kernel_func) {
        throw std::runtime_error("adam_step_cu kernel function pointer is null!");
    }

    // Get device properties to validate launch config
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("[adam_step] Device %d: %s, maxThreadsPerBlock=%d, maxGridSize=[%d,%d,%d]\n",
           device, props.name, props.maxThreadsPerBlock,
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    fflush(stdout);

    if (block_size > props.maxThreadsPerBlock) {
        throw std::runtime_error("block_size exceeds maxThreadsPerBlock");
    }

    kernels::adam::adam_step_cu<<<grid_size, block_size>>>(
        param,
        exp_avg,
        exp_avg_sq,
        param_grad,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);

    // Always check for errors to ensure kernel launched successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("adam_step_cu kernel launch failed: ") + cudaGetErrorString(err));
    }

    CHECK_CUDA(config::debug, "adam step")
}
