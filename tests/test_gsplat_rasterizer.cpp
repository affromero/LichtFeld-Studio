/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

namespace {
    constexpr uint32_t kDepthGradN = 1;
    constexpr uint32_t kDepthGradC = 1;
    constexpr uint32_t kDepthGradK = 1;
    constexpr uint32_t kDepthGradW = 32;
    constexpr uint32_t kDepthGradH = 32;
    constexpr uint32_t kDepthGradTileSize = 16;
    constexpr uint32_t kDepthGradChannels = 4;
    constexpr int kDepthGradRenderModeRgbExpectedDepth = 4;

    void assert_cuda_ok(cudaError_t error, const char* context) {
        ASSERT_EQ(error, cudaSuccess)
            << context << ": " << cudaGetErrorString(error);
    }

    template <typename T>
    class DeviceBuffer {
    public:
        DeviceBuffer() = default;

        explicit DeviceBuffer(size_t count)
            : count_(count) {
            if (count_ == 0) {
                return;
            }
            const cudaError_t error = cudaMalloc(&ptr_, count_ * sizeof(T));
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMalloc failed: ") +
                    cudaGetErrorString(error));
            }
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept
            : ptr_(other.ptr_),
              count_(other.count_) {
            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                release();
                ptr_ = other.ptr_;
                count_ = other.count_;
                other.ptr_ = nullptr;
                other.count_ = 0;
            }
            return *this;
        }

        ~DeviceBuffer() {
            release();
        }

        T* ptr() const {
            return ptr_;
        }

        size_t count() const {
            return count_;
        }

        void zero() const {
            if (ptr_ == nullptr) {
                return;
            }
            const cudaError_t error =
                cudaMemset(ptr_, 0, count_ * sizeof(T));
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemset failed: ") +
                    cudaGetErrorString(error));
            }
        }

        void copy_from(const std::vector<T>& values) const {
            if (values.size() != count_) {
                throw std::runtime_error("host/device copy size mismatch");
            }
            const cudaError_t error = cudaMemcpy(
                ptr_,
                values.data(),
                count_ * sizeof(T),
                cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemcpy H2D failed: ") +
                    cudaGetErrorString(error));
            }
        }

        std::vector<T> copy_to_host() const {
            std::vector<T> values(count_);
            const cudaError_t error = cudaMemcpy(
                values.data(),
                ptr_,
                count_ * sizeof(T),
                cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemcpy D2H failed: ") +
                    cudaGetErrorString(error));
            }
            return values;
        }

    private:
        void release() {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
                ptr_ = nullptr;
            }
            count_ = 0;
        }

        T* ptr_ = nullptr;
        size_t count_ = 0;
    };

    DeviceBuffer<float> cuda_float_buffer(const std::vector<float>& values) {
        DeviceBuffer<float> buffer(values.size());
        buffer.copy_from(values);
        return buffer;
    }

    DeviceBuffer<float> cuda_zero_float_buffer(size_t count) {
        DeviceBuffer<float> buffer(count);
        buffer.zero();
        return buffer;
    }

    DeviceBuffer<int32_t> cuda_zero_int_buffer(size_t count) {
        DeviceBuffer<int32_t> buffer(count);
        buffer.zero();
        return buffer;
    }

    struct DepthGradientScene {
        DeviceBuffer<float> means;
        DeviceBuffer<float> quats;
        DeviceBuffer<float> scales;
        DeviceBuffer<float> opacities;
        DeviceBuffer<float> sh_coeffs;
        DeviceBuffer<float> viewmats;
        DeviceBuffer<float> intrinsics;
    };

    struct DepthForwardBuffers {
        DeviceBuffer<float> render_colors;
        DeviceBuffer<float> render_alphas;
        DeviceBuffer<int32_t> radii;
        DeviceBuffer<float> means2d;
        DeviceBuffer<float> depths;
        DeviceBuffer<float> colors;
        DeviceBuffer<float> rgb_colors;
        DeviceBuffer<float> dirs;
        DeviceBuffer<float> conics;
        DeviceBuffer<int32_t> tiles_per_gauss;
        DeviceBuffer<int32_t> tile_offsets;
        DeviceBuffer<int32_t> last_ids;
        int64_t* isect_ids = nullptr;
        int32_t* flatten_ids = nullptr;
        int32_t n_isects = 0;

        DepthForwardBuffers() = default;
        DepthForwardBuffers(const DepthForwardBuffers&) = delete;
        DepthForwardBuffers& operator=(const DepthForwardBuffers&) = delete;

        DepthForwardBuffers(DepthForwardBuffers&& other) noexcept
            : render_colors(std::move(other.render_colors)),
              render_alphas(std::move(other.render_alphas)),
              radii(std::move(other.radii)),
              means2d(std::move(other.means2d)),
              depths(std::move(other.depths)),
              colors(std::move(other.colors)),
              rgb_colors(std::move(other.rgb_colors)),
              dirs(std::move(other.dirs)),
              conics(std::move(other.conics)),
              tiles_per_gauss(std::move(other.tiles_per_gauss)),
              tile_offsets(std::move(other.tile_offsets)),
              last_ids(std::move(other.last_ids)),
              isect_ids(other.isect_ids),
              flatten_ids(other.flatten_ids),
              n_isects(other.n_isects) {
            other.isect_ids = nullptr;
            other.flatten_ids = nullptr;
            other.n_isects = 0;
        }

        DepthForwardBuffers& operator=(DepthForwardBuffers&& other) noexcept {
            if (this != &other) {
                release_intersections();
                render_colors = std::move(other.render_colors);
                render_alphas = std::move(other.render_alphas);
                radii = std::move(other.radii);
                means2d = std::move(other.means2d);
                depths = std::move(other.depths);
                colors = std::move(other.colors);
                rgb_colors = std::move(other.rgb_colors);
                dirs = std::move(other.dirs);
                conics = std::move(other.conics);
                tiles_per_gauss = std::move(other.tiles_per_gauss);
                tile_offsets = std::move(other.tile_offsets);
                last_ids = std::move(other.last_ids);
                isect_ids = other.isect_ids;
                flatten_ids = other.flatten_ids;
                n_isects = other.n_isects;
                other.isect_ids = nullptr;
                other.flatten_ids = nullptr;
                other.n_isects = 0;
            }
            return *this;
        }

        ~DepthForwardBuffers() {
            release_intersections();
        }

        void release_intersections() {
            if (isect_ids != nullptr) {
                cudaFree(isect_ids);
                isect_ids = nullptr;
            }
            if (flatten_ids != nullptr) {
                cudaFree(flatten_ids);
                flatten_ids = nullptr;
            }
            n_isects = 0;
        }
    };

    DepthGradientScene make_depth_gradient_scene() {
        return DepthGradientScene{
            .means = cuda_float_buffer({0.0f, 0.0f, 3.0f}),
            .quats = cuda_float_buffer({1.0f, 0.0f, 0.0f, 0.0f}),
            .scales = cuda_float_buffer({0.35f, 0.35f, 0.35f}),
            .opacities = cuda_float_buffer({0.95f}),
            .sh_coeffs = cuda_float_buffer({0.25f, 0.35f, 0.45f}),
            .viewmats = cuda_float_buffer(
                {1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 1.0f}),
            .intrinsics = cuda_float_buffer(
                {30.0f, 0.0f, 15.5f,
                 0.0f, 30.0f, 15.5f,
                 0.0f, 0.0f, 1.0f})};
    }

    DepthForwardBuffers make_depth_forward_buffers() {
        const uint32_t tile_width =
            (kDepthGradW + kDepthGradTileSize - 1) / kDepthGradTileSize;
        const uint32_t tile_height =
            (kDepthGradH + kDepthGradTileSize - 1) / kDepthGradTileSize;
        DepthForwardBuffers buffers;
        buffers.render_colors = cuda_zero_float_buffer(
            kDepthGradC * kDepthGradH * kDepthGradW * kDepthGradChannels);
        buffers.render_alphas = cuda_zero_float_buffer(
            kDepthGradC * kDepthGradH * kDepthGradW);
        buffers.radii = cuda_zero_int_buffer(kDepthGradC * kDepthGradN * 2);
        buffers.means2d = cuda_zero_float_buffer(kDepthGradC * kDepthGradN * 2);
        buffers.depths = cuda_zero_float_buffer(kDepthGradC * kDepthGradN);
        buffers.colors = cuda_zero_float_buffer(
            kDepthGradC * kDepthGradN * kDepthGradChannels);
        buffers.rgb_colors = cuda_zero_float_buffer(kDepthGradC * kDepthGradN * 3);
        buffers.dirs = cuda_zero_float_buffer(kDepthGradC * kDepthGradN * 3);
        buffers.conics = cuda_zero_float_buffer(kDepthGradC * kDepthGradN * 3);
        buffers.tiles_per_gauss = cuda_zero_int_buffer(kDepthGradC * kDepthGradN);
        buffers.tile_offsets = cuda_zero_int_buffer(kDepthGradC * tile_height * tile_width);
        buffers.last_ids = cuda_zero_int_buffer(kDepthGradC * kDepthGradH * kDepthGradW);
        return buffers;
    }

    DepthForwardBuffers run_depth_forward(const DepthGradientScene& scene) {
        auto buffers = make_depth_forward_buffers();
        gsplat_lfs::RasterizeWithSHResult result{
            .render_colors = buffers.render_colors.ptr(),
            .render_alphas = buffers.render_alphas.ptr(),
            .radii = buffers.radii.ptr(),
            .means2d = buffers.means2d.ptr(),
            .depths = buffers.depths.ptr(),
            .colors = buffers.colors.ptr(),
            .rgb_colors = buffers.rgb_colors.ptr(),
            .dirs = buffers.dirs.ptr(),
            .conics = buffers.conics.ptr(),
            .tiles_per_gauss = buffers.tiles_per_gauss.ptr(),
            .tile_offsets = buffers.tile_offsets.ptr(),
            .last_ids = buffers.last_ids.ptr(),
            .compensations = nullptr,
            .isect_ids = nullptr,
            .flatten_ids = nullptr,
            .n_isects = 0};

        UnscentedTransformParameters ut_params;
        gsplat_lfs::rasterize_from_world_with_sh_fwd(
            scene.means.ptr(),
            scene.quats.ptr(),
            scene.scales.ptr(),
            scene.opacities.ptr(),
            scene.sh_coeffs.ptr(),
            0,
            nullptr,
            nullptr,
            nullptr,
            kDepthGradN,
            kDepthGradC,
            kDepthGradK,
            kDepthGradW,
            kDepthGradH,
            kDepthGradTileSize,
            scene.viewmats.ptr(),
            nullptr,
            scene.intrinsics.ptr(),
            PINHOLE,
            0.3f,
            0.01f,
            100.0f,
            0.0f,
            1.0f,
            false,
            kDepthGradRenderModeRgbExpectedDepth,
            ut_params,
            ShutterType::GLOBAL,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            result,
            nullptr);
        assert_cuda_ok(cudaDeviceSynchronize(), "depth forward");

        buffers.isect_ids = result.isect_ids;
        buffers.flatten_ids = result.flatten_ids;
        buffers.n_isects = result.n_isects;
        return buffers;
    }

    double expected_depth_loss(
        const DepthForwardBuffers& buffers,
        const std::vector<uint8_t>& selected_pixels) {
        const std::vector<float> colors = buffers.render_colors.copy_to_host();
        const std::vector<float> alphas = buffers.render_alphas.copy_to_host();
        double loss = 0.0;
        for (size_t index = 0; index < selected_pixels.size(); ++index) {
            if (selected_pixels[index] == 0) {
                continue;
            }
            const double alpha = static_cast<double>(alphas[index]);
            const double accum_depth =
                static_cast<double>(colors[index * kDepthGradChannels + 3]);
            loss += accum_depth / alpha;
        }
        return loss;
    }

    std::vector<uint8_t> selected_depth_pixels(const DepthForwardBuffers& buffers) {
        const std::vector<float> alphas = buffers.render_alphas.copy_to_host();
        std::vector<uint8_t> selected(kDepthGradH * kDepthGradW, 0);
        size_t selected_count = 0;
        for (size_t index = 0; index < selected.size(); ++index) {
            if (alphas[index] > 1e-3f) {
                selected[index] = 1;
                selected_count++;
            }
        }
        EXPECT_GT(selected_count, 0UL);
        return selected;
    }

    void fill_expected_depth_upstream_grads(
        const DepthForwardBuffers& buffers,
        const std::vector<uint8_t>& selected_pixels,
        std::vector<float>* v_render_colors,
        std::vector<float>* v_render_alphas) {
        const std::vector<float> colors = buffers.render_colors.copy_to_host();
        const std::vector<float> alphas = buffers.render_alphas.copy_to_host();

        v_render_colors->assign(
            kDepthGradH * kDepthGradW * kDepthGradChannels,
            0.0f);
        v_render_alphas->assign(kDepthGradH * kDepthGradW, 0.0f);
        for (size_t index = 0; index < selected_pixels.size(); ++index) {
            if (selected_pixels[index] == 0) {
                continue;
            }
            const float alpha = alphas[index];
            const float accum_depth = colors[index * kDepthGradChannels + 3];
            const float expected_depth = accum_depth / alpha;
            (*v_render_colors)[index * kDepthGradChannels + 3] = 1.0f / alpha;
            (*v_render_alphas)[index] = -expected_depth / alpha;
        }
    }

} // namespace

class GsplatRasterizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create minimal test data
        const size_t N = 100; // Number of Gaussians
        const int sh_degree = 0;

        // Create random Gaussian parameters
        means_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32);
        sh0_ = Tensor::randn({N, 1, 3}, Device::CUDA, DataType::Float32);            // sh0 is [N, 1, 3]
        shN_ = Tensor::zeros({N, 0, 3}, Device::CUDA, DataType::Float32);            // No higher SH for degree 0
        scaling_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32).mul(0.1f); // Small scales
        rotation_ = Tensor::randn({N, 4}, Device::CUDA, DataType::Float32);
        opacity_ = Tensor::randn({N}, Device::CUDA, DataType::Float32);

        // Create SplatData
        splat_data_ = std::make_unique<SplatData>(
            sh_degree,
            means_,
            sh0_,
            shN_,
            scaling_,
            rotation_,
            opacity_,
            1.0f // scene_scale
        );

        // Create camera
        auto R = Tensor::eye(3, Device::CUDA);
        auto T = Tensor::zeros({3}, Device::CUDA, DataType::Float32);

        // Set camera at z=3 looking at origin
        std::vector<float> T_data = {0.0f, 0.0f, 3.0f};
        T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        camera_ = std::make_unique<Camera>(
            R, T,
            500.0f, 500.0f, // focal_x, focal_y
            320.0f, 240.0f, // center_x, center_y
            Tensor(),       // radial_distortion
            Tensor(),       // tangential_distortion
            lfs::core::CameraModelType::PINHOLE,
            "test_image",
            std::filesystem::path{}, // image_path
            std::filesystem::path{}, // mask_path
            std::filesystem::path{}, // depth_path
            640, 480,                // camera_width, camera_height (constructor sets image_width/height too)
            0                        // uid
        );

        // Background color
        bg_color_ = Tensor::zeros({3}, Device::CUDA, DataType::Float32);
        bg_color_.fill_(0.5f); // Gray background
    }

    std::unique_ptr<SplatData> splat_data_;
    std::unique_ptr<Camera> camera_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_;
    Tensor bg_color_;
};

TEST_F(GsplatRasterizerTest, ForwardPassBasic) {
    // Just test that forward pass doesn't crash
    auto result = gsplat_rasterize_forward(
        *camera_, *splat_data_, bg_color_,
        0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB);

    ASSERT_TRUE(result.has_value()) << "Forward pass failed: " << result.error();

    auto& [render_output, ctx] = result.value();

    // Check output dimensions
    EXPECT_EQ(render_output.width, 640);
    EXPECT_EQ(render_output.height, 480);
    EXPECT_TRUE(render_output.image.is_valid());
    EXPECT_EQ(render_output.image.shape()[0], 3); // CHW format
    EXPECT_EQ(render_output.image.shape()[1], 480);
    EXPECT_EQ(render_output.image.shape()[2], 640);

    // Check alpha
    EXPECT_TRUE(render_output.alpha.is_valid());
    EXPECT_EQ(render_output.alpha.shape()[0], 1);
    EXPECT_EQ(render_output.alpha.shape()[1], 480);
    EXPECT_EQ(render_output.alpha.shape()[2], 640);

    std::cout << "Forward pass succeeded!" << std::endl;
    std::cout << "  Image shape: [" << render_output.image.shape()[0] << ", "
              << render_output.image.shape()[1] << ", "
              << render_output.image.shape()[2] << "]" << std::endl;
}

TEST_F(GsplatRasterizerTest, InferenceWrapper) {
    // Test the convenience wrapper
    EXPECT_NO_THROW({
        auto output = gsplat_rasterize(*camera_, *splat_data_, bg_color_);
        EXPECT_TRUE(output.image.is_valid());
    });
}

TEST(GsplatDepthGradientTest, RgbExpectedDepthMatchesFiniteDifferenceForMeanZ) {
    int cuda_device_count = 0;
    const cudaError_t device_error = cudaGetDeviceCount(&cuda_device_count);
    if (device_error != cudaSuccess || cuda_device_count == 0) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto scene = make_depth_gradient_scene();
    auto base_forward = run_depth_forward(scene);
    ASSERT_GT(base_forward.n_isects, 0);
    const std::vector<uint8_t> selected_pixels =
        selected_depth_pixels(base_forward);

    std::vector<float> v_render_colors_host;
    std::vector<float> v_render_alphas_host;
    fill_expected_depth_upstream_grads(
        base_forward,
        selected_pixels,
        &v_render_colors_host,
        &v_render_alphas_host);

    auto v_render_colors = cuda_float_buffer(v_render_colors_host);
    auto v_render_alphas = cuda_float_buffer(v_render_alphas_host);
    auto v_means = cuda_zero_float_buffer(kDepthGradN * 3);
    auto v_quats = cuda_zero_float_buffer(kDepthGradN * 4);
    auto v_scales = cuda_zero_float_buffer(kDepthGradN * 3);
    auto v_opacities = cuda_zero_float_buffer(kDepthGradN);
    auto v_sh_coeffs = cuda_zero_float_buffer(kDepthGradN * kDepthGradK * 3);

    UnscentedTransformParameters ut_params;
    gsplat_lfs::rasterize_from_world_with_sh_bwd(
        scene.means.ptr(),
        scene.quats.ptr(),
        scene.scales.ptr(),
        scene.opacities.ptr(),
        scene.sh_coeffs.ptr(),
        0,
        nullptr,
        nullptr,
        nullptr,
        kDepthGradN,
        kDepthGradC,
        kDepthGradK,
        kDepthGradW,
        kDepthGradH,
        kDepthGradTileSize,
        scene.viewmats.ptr(),
        nullptr,
        scene.intrinsics.ptr(),
        PINHOLE,
        0.3f,
        0.01f,
        100.0f,
        0.0f,
        1.0f,
        false,
        kDepthGradRenderModeRgbExpectedDepth,
        ut_params,
        ShutterType::GLOBAL,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        base_forward.render_alphas.ptr(),
        base_forward.last_ids.ptr(),
        base_forward.tile_offsets.ptr(),
        base_forward.flatten_ids,
        static_cast<uint32_t>(base_forward.n_isects),
        base_forward.colors.ptr(),
        base_forward.radii.ptr(),
        base_forward.means2d.ptr(),
        base_forward.depths.ptr(),
        nullptr,
        v_render_colors.ptr(),
        v_render_alphas.ptr(),
        v_means.ptr(),
        v_quats.ptr(),
        v_scales.ptr(),
        v_opacities.ptr(),
        v_sh_coeffs.ptr(),
        nullptr,
        nullptr,
        nullptr);
    assert_cuda_ok(cudaDeviceSynchronize(), "depth backward");

    const std::vector<float> analytic_grad = v_means.copy_to_host();
    const double analytic_z = static_cast<double>(analytic_grad[2]);

    constexpr float epsilon = 1e-3f;
    auto plus_scene = make_depth_gradient_scene();
    auto minus_scene = make_depth_gradient_scene();
    std::vector<float> plus_mean = {0.0f, 0.0f, 3.0f + epsilon};
    std::vector<float> minus_mean = {0.0f, 0.0f, 3.0f - epsilon};
    assert_cuda_ok(
        cudaMemcpy(
            plus_scene.means.ptr(),
            plus_mean.data(),
            plus_mean.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy plus mean");
    assert_cuda_ok(
        cudaMemcpy(
            minus_scene.means.ptr(),
            minus_mean.data(),
            minus_mean.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy minus mean");

    auto plus_forward = run_depth_forward(plus_scene);
    auto minus_forward = run_depth_forward(minus_scene);
    const double plus_loss = expected_depth_loss(plus_forward, selected_pixels);
    const double minus_loss = expected_depth_loss(minus_forward, selected_pixels);
    const double finite_diff_z =
        (plus_loss - minus_loss) / (2.0 * static_cast<double>(epsilon));

    const double abs_error = std::abs(analytic_z - finite_diff_z);
    const double rel_error =
        abs_error / std::max(1e-6, std::abs(finite_diff_z));

    EXPECT_LT(rel_error, 0.10)
        << "RGB_ED depth gradient for mean.z disagrees with finite difference. "
        << "analytic=" << analytic_z
        << ", finite_diff=" << finite_diff_z
        << ", abs_error=" << abs_error
        << ", rel_error=" << rel_error;
}
