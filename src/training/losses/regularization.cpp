/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "lfs/kernels/regularization.cuh" // LibTorch-free CUDA kernels
#include <format>

namespace lfs::training::losses {

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward(
        const lfs::core::Tensor& scaling_raw,
        lfs::core::Tensor& scaling_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Validate inputs
            if (scaling_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw must be on CUDA device");
            }
            if (scaling_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw_grad must be on CUDA device");
            }
            if (scaling_raw.shape() != scaling_raw_grad.shape()) {
                return std::unexpected("scaling_raw and scaling_raw_grad must have same shape");
            }

            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_scale_regularization(
                scaling_raw.ptr<float>(),
                scaling_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward(
        const lfs::core::Tensor& opacity_raw,
        lfs::core::Tensor& opacity_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Validate inputs
            if (opacity_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw must be on CUDA device");
            }
            if (opacity_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw_grad must be on CUDA device");
            }
            if (opacity_raw.shape() != opacity_raw_grad.shape()) {
                return std::unexpected("opacity_raw and opacity_raw_grad must have same shape");
            }

            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_opacity_regularization(
                opacity_raw.ptr<float>(),
                opacity_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> PositionRegularization::forward(
        const lfs::core::Tensor& means,
        const lfs::core::Tensor& init_means,
        lfs::core::Tensor& means_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Validate inputs
            if (means.device() != lfs::core::Device::CUDA) {
                return std::unexpected("means must be on CUDA device");
            }
            if (init_means.device() != lfs::core::Device::CUDA) {
                return std::unexpected("init_means must be on CUDA device");
            }
            if (means_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("means_grad must be on CUDA device");
            }
            if (means.shape() != init_means.shape()) {
                return std::unexpected("means and init_means must have same shape");
            }
            if (means.shape() != means_grad.shape()) {
                return std::unexpected("means and means_grad must have same shape");
            }

            size_t n = means.shape()[0];
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Compute diff = means - init_means
            auto diff = means.sub(init_means);

            // Compute squared norm per point: ||diff||² = sum(diff * diff, dim=1)
            auto diff_sq = diff.mul(diff);
            auto sq_norms = diff_sq.sum(1);  // [N] - sum over xyz dimension

            // Loss = weight * mean(sq_norms)
            auto loss_tensor = sq_norms.mean().mul(params.weight);

            // Gradient: ∂L/∂means = (2 * weight / N) * diff
            float grad_scale = 2.0f * params.weight / static_cast<float>(n);
            auto grad_contribution = diff.mul(grad_scale);

            // Accumulate gradients
            means_grad.add_(grad_contribution);

            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in PositionRegularization::forward: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
