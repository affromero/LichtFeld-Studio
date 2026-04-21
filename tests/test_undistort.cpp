/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "io/formats/colmap.hpp"
#include "training/rasterization/gsplat/Cameras.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

using namespace lfs::core;

namespace {

    constexpr float TEST_FX = 500.0f;
    constexpr float TEST_FY = 500.0f;
    constexpr float TEST_CX = 320.0f;
    constexpr float TEST_CY = 240.0f;
    constexpr int TEST_W = 640;
    constexpr int TEST_H = 480;

    void validate_params(const UndistortParams& p, int src_w, int src_h) {
        EXPECT_GT(p.dst_width, 0);
        EXPECT_GT(p.dst_height, 0);
        EXPECT_LE(p.dst_width, src_w * 2);
        EXPECT_LE(p.dst_height, src_h * 2);
        EXPECT_GT(p.dst_fx, 0.0f);
        EXPECT_GT(p.dst_fy, 0.0f);
        EXPECT_GT(p.dst_cx, 0.0f);
        EXPECT_GT(p.dst_cy, 0.0f);
    }

    void run_image_undistort(const UndistortParams& params) {
        auto src = Tensor::randn(
            {3, static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
            Device::CUDA);

        auto dst = undistort_image(src, params, nullptr);
        cudaDeviceSynchronize();

        ASSERT_EQ(dst.ndim(), 3u);
        EXPECT_EQ(static_cast<int>(dst.shape()[0]), 3);
        EXPECT_EQ(static_cast<int>(dst.shape()[1]), params.dst_height);
        EXPECT_EQ(static_cast<int>(dst.shape()[2]), params.dst_width);
    }

    void run_mask_undistort(const UndistortParams& params) {
        auto src = Tensor::ones(
            {static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
            Device::CUDA);

        auto dst = undistort_mask(src, params, nullptr);
        cudaDeviceSynchronize();

        ASSERT_EQ(dst.ndim(), 2u);
        EXPECT_EQ(static_cast<int>(dst.shape()[0]), params.dst_height);
        EXPECT_EQ(static_cast<int>(dst.shape()[1]), params.dst_width);
    }

    glm::fvec2 project_rational_reference(
        const glm::fvec3& cam_ray,
        const std::array<float, 8>& rational_coeffs,
        const std::array<float, 3>& tangential_coeffs,
        float fx,
        float fy,
        float cx,
        float cy) {
        const float xn = cam_ray.x / cam_ray.z;
        const float yn = cam_ray.y / cam_ray.z;
        const float r2 = xn * xn + yn * yn;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;

        const float b_num =
            1.0f + rational_coeffs[0] * r2 +
            rational_coeffs[1] * r4 +
            rational_coeffs[2] * r6;
        const float b_den =
            1.0f + rational_coeffs[3] * r2 +
            rational_coeffs[4] * r4 +
            rational_coeffs[5] * r6;
        const float safe_den =
            (b_den < 0.0f ? -1.0f : 1.0f) * std::max(std::fabs(b_den), 1e-12f);
        const float B = b_num / safe_den;
        const float A = 1.0f + rational_coeffs[6] * r2 + rational_coeffs[7] * r4;
        const float p1 = tangential_coeffs[0];
        const float p2 = tangential_coeffs[1];
        const float skew = tangential_coeffs[2];

        const float xd =
            xn * B +
            A * (p2 * (r2 + 2.0f * xn * xn) + 2.0f * p1 * xn * yn);
        const float yd =
            yn * B +
            A * (p1 * (r2 + 2.0f * yn * yn) + 2.0f * p2 * xn * yn);

        return {
            fx * xd + skew * yd + cx,
            fy * yd + cy,
        };
    }

} // namespace

// ====================== Coefficient packing tests ======================

TEST(UndistortPacking, PinholeRadialOnly) {
    // COLMAP SIMPLE_RADIAL / RADIAL: 1-2 radial, no tangential
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f); // k3 = 0
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f); // p1 = 0
    EXPECT_FLOAT_EQ(params.distortion[4], 0.0f); // p2 = 0
}

TEST(UndistortPacking, PinholeRadialAndTangential) {
    // COLMAP OPENCV: 2 radial (k1,k2) + 2 tangential (p1,p2)
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);   // k1
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);    // k3 = 0
    EXPECT_FLOAT_EQ(params.distortion[3], 0.003f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[4], -0.004f); // p2
    EXPECT_EQ(params.num_distortion, 5);
}

TEST(UndistortPacking, PinholeFullRadialAndTangential) {
    // COLMAP FULL_OPENCV: 6 radial + 2 tangential (only 3 radial used by our kernel)
    auto radial = Tensor::from_vector({-0.1f, 0.02f, -0.003f}, TensorShape({3}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.002f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);   // k1
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], -0.003f); // k3
    EXPECT_FLOAT_EQ(params.distortion[3], 0.001f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[4], -0.002f); // p2
    EXPECT_EQ(params.num_distortion, 5);
}

TEST(UndistortPacking, Fisheye4Coeffs) {
    // COLMAP OPENCV_FISHEYE: 4 radial (k1-k4), no tangential
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.005f, -0.001f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.1f);
    EXPECT_FLOAT_EQ(params.distortion[1], -0.02f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.005f);
    EXPECT_FLOAT_EQ(params.distortion[3], -0.001f);
    EXPECT_EQ(params.num_distortion, 4);
}

TEST(UndistortPacking, Fisheye1Coeff) {
    // COLMAP SIMPLE_RADIAL_FISHEYE: 1 radial
    auto radial = Tensor::from_vector({0.05f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.05f);
    EXPECT_FLOAT_EQ(params.distortion[1], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f);
    EXPECT_EQ(params.num_distortion, 1);
}

TEST(UndistortPacking, Fisheye2Coeffs) {
    // COLMAP RADIAL_FISHEYE: 2 radial (k1, k2)
    auto radial = Tensor::from_vector({0.05f, -0.01f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.05f);
    EXPECT_FLOAT_EQ(params.distortion[1], -0.01f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f);
    EXPECT_EQ(params.num_distortion, 2);
}

TEST(UndistortPacking, ThinPrismFisheye) {
    // COLMAP THIN_PRISM_FISHEYE: radial={k1,k2,k3,k4}, tangential={p1,p2,s1,s2}
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.001f}, TensorShape({4}), Device::CPU);
    auto tangential = Tensor::from_vector({0.0005f, -0.0003f, 0.0001f, -0.0002f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.1f);     // k1
    EXPECT_FLOAT_EQ(params.distortion[1], -0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], 0.003f);   // k3
    EXPECT_FLOAT_EQ(params.distortion[3], -0.001f);  // k4
    EXPECT_FLOAT_EQ(params.distortion[4], 0.0005f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[5], -0.0003f); // p2
    EXPECT_FLOAT_EQ(params.distortion[6], 0.0001f);  // s1
    EXPECT_FLOAT_EQ(params.distortion[7], -0.0002f); // s2
    EXPECT_EQ(params.num_distortion, 8);
}

TEST(UndistortPacking, Rational) {
    auto radial = Tensor::from_vector(
        {0.11f, -0.03f, 0.004f, 0.02f, -0.005f, 0.0008f, 0.01f, -0.002f},
        TensorShape({8}), Device::CPU);
    auto tangential = Tensor::from_vector(
        {0.0006f, -0.0004f, 0.75f},
        TensorShape({3}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::RATIONAL);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.11f);      // b1
    EXPECT_FLOAT_EQ(params.distortion[1], -0.03f);     // b2
    EXPECT_FLOAT_EQ(params.distortion[2], 0.004f);     // b3
    EXPECT_FLOAT_EQ(params.distortion[3], 0.02f);      // d1
    EXPECT_FLOAT_EQ(params.distortion[4], -0.005f);    // d2
    EXPECT_FLOAT_EQ(params.distortion[5], 0.0008f);    // d3
    EXPECT_FLOAT_EQ(params.distortion[6], 0.01f);      // a1
    EXPECT_FLOAT_EQ(params.distortion[7], -0.002f);    // a2
    EXPECT_FLOAT_EQ(params.distortion[8], 0.0006f);    // p1
    EXPECT_FLOAT_EQ(params.distortion[9], -0.0004f);   // p2
    EXPECT_FLOAT_EQ(params.distortion[10], 0.75f);     // skew
    EXPECT_EQ(params.num_distortion, 11);
}

TEST(RationalCameraModel, ForwardProjectionMatchesReference) {
    RationalCameraModel<>::Parameters params = {};
    params.resolution = {TEST_W, TEST_H};
    params.shutter_type = ShutterType::GLOBAL;
    params.principal_point = {319.5f, 241.25f};
    params.focal_length = {512.0f, 505.0f};
    params.rational_coeffs = {
        0.12f, -0.04f, 0.006f,
        0.02f, -0.003f, 0.0005f,
        0.015f, -0.002f};
    params.tangential_coeffs = {0.0008f, -0.0003f, 0.9f};

    RationalCameraModel<> camera_model(params);
    const glm::fvec3 cam_ray{0.35f, -0.18f, 1.4f};
    const auto [image_point, valid] =
        camera_model.camera_ray_to_image_point(cam_ray, 0.0f);
    const auto expected = project_rational_reference(
        cam_ray,
        params.rational_coeffs,
        params.tangential_coeffs,
        params.focal_length[0],
        params.focal_length[1],
        params.principal_point[0],
        params.principal_point[1]);

    EXPECT_TRUE(valid);
    EXPECT_NEAR(image_point.x, expected.x, 1e-5f);
    EXPECT_NEAR(image_point.y, expected.y, 1e-5f);
}

TEST(RationalCameraModel, Scan003ForwardProjectionMatchesReference) {
    RationalCameraModel<>::Parameters params = {};
    params.resolution = {728, 544};
    params.shutter_type = ShutterType::GLOBAL;
    params.principal_point = {362.5429993f, 283.7081604f};
    params.focal_length = {317.8523280f, 317.7174315f};
    params.rational_coeffs = {
        0.3415374160f, 0.0f, 0.0f,
        0.7019518018f, 0.0478103422f, -0.0008384892f,
        -0.2079813331f, 0.0461935438f};
    params.tangential_coeffs = {
        -0.0000266082f, 0.0000096041f, 0.0415059291f};

    RationalCameraModel<> camera_model(params);
    const std::array<glm::fvec3, 5> cam_rays = {{
        {0.0f, 0.0f, 1.0f},
        {0.2f, -0.1f, 1.0f},
        {-0.75f, 0.35f, 1.0f},
        {3.25f, -2.0f, 1.0f},
        {-4.0f, 3.0f, 1.0f},
    }};

    for (const auto& cam_ray : cam_rays) {
        const auto [image_point, valid] =
            camera_model.camera_ray_to_image_point(cam_ray, 0.0f);
        const auto expected = project_rational_reference(
            cam_ray,
            params.rational_coeffs,
            params.tangential_coeffs,
            params.focal_length[0],
            params.focal_length[1],
            params.principal_point[0],
            params.principal_point[1]);

        EXPECT_TRUE(valid);
        EXPECT_NEAR(image_point.x, expected.x, 1e-4f);
        EXPECT_NEAR(image_point.y, expected.y, 1e-4f);
    }
}

TEST(RationalCameraModel, Scan003NewtonInverseRoundTripsPixels) {
    RationalCameraModel<>::Parameters params = {};
    params.resolution = {728, 544};
    params.shutter_type = ShutterType::GLOBAL;
    params.principal_point = {362.5429993f, 283.7081604f};
    params.focal_length = {317.8523280f, 317.7174315f};
    params.rational_coeffs = {
        0.3415374160f, 0.0f, 0.0f,
        0.7019518018f, 0.0478103422f, -0.0008384892f,
        -0.2079813331f, 0.0461935438f};
    params.tangential_coeffs = {
        -0.0000266082f, 0.0000096041f, 0.0415059291f};

    RationalCameraModel<> camera_model(params);
    const std::array<glm::fvec2, 5> image_points = {{
        {0.5f, 0.5f},
        {727.5f, 543.5f},
        {362.5f, 283.5f},
        {40.0f, 500.0f},
        {690.0f, 25.0f},
    }};

    for (const auto& image_point : image_points) {
        const auto [cam_ray, valid] =
            camera_model.image_point_to_camera_ray(image_point);
        ASSERT_TRUE(valid);

        const auto [reprojected, reproj_valid] =
            camera_model.camera_ray_to_image_point(cam_ray, 0.0f);
        EXPECT_TRUE(reproj_valid);
        EXPECT_NEAR(reprojected.x, image_point.x, 1e-3f);
        EXPECT_NEAR(reprojected.y, image_point.y, 1e-3f);
    }
}

TEST(CameraRawRationalLayout, RotatesTransposedImageBackToSensorFrame) {
    const auto rotation = Tensor::eye(3, Device::CPU);
    const auto translation = Tensor::zeros({3}, Device::CPU);
    const auto rational = Tensor::zeros({8}, Device::CPU);
    const auto tangential = Tensor::zeros({3}, Device::CPU);

    Camera camera(
        rotation,
        translation,
        100.0f,
        100.0f,
        1.0f,
        1.0f,
        rational,
        tangential,
        CameraModelType::RATIONAL,
        "raw.png",
        {},
        {},
        3,
        2,
        0);

    const auto image = Tensor::from_vector(
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        TensorShape({1, 3, 2}),
        Device::CPU);
    const auto rotated =
        camera.normalize_loaded_image_orientation(image).cpu();

    ASSERT_EQ(rotated.shape()[0], 1u);
    ASSERT_EQ(rotated.shape()[1], 2u);
    ASSERT_EQ(rotated.shape()[2], 3u);

    const auto rotated_acc = rotated.accessor<float, 3>();
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 1), 4.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 2), 6.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 1, 0), 1.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 1, 2), 5.0f);

    const auto mask = Tensor::from_vector(
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        TensorShape({3, 2}),
        Device::CPU);
    const auto rotated_mask =
        camera.normalize_loaded_mask_orientation(mask).cpu();
    const auto mask_acc = rotated_mask.accessor<float, 2>();
    ASSERT_EQ(rotated_mask.shape()[0], 2u);
    ASSERT_EQ(rotated_mask.shape()[1], 3u);
    EXPECT_FLOAT_EQ(mask_acc(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(mask_acc(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(mask_acc(0, 2), 6.0f);
    EXPECT_FLOAT_EQ(mask_acc(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask_acc(1, 1), 3.0f);
    EXPECT_FLOAT_EQ(mask_acc(1, 2), 5.0f);
}

TEST(CameraRawRationalLayout, RotatesDownsampledTransposedImageBackToSensorFrame) {
    const auto rotation = Tensor::eye(3, Device::CPU);
    const auto translation = Tensor::zeros({3}, Device::CPU);
    const auto rational = Tensor::zeros({8}, Device::CPU);
    const auto tangential = Tensor::zeros({3}, Device::CPU);

    Camera camera(
        rotation,
        translation,
        100.0f,
        100.0f,
        1.0f,
        1.0f,
        rational,
        tangential,
        CameraModelType::RATIONAL,
        "raw_downsampled.png",
        {},
        {},
        8,
        6,
        0);

    const auto image = Tensor::from_vector(
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         10.0f, 11.0f, 12.0f},
        TensorShape({1, 4, 3}),
        Device::CPU);
    const auto rotated =
        camera.normalize_loaded_image_orientation(image).cpu();

    ASSERT_EQ(rotated.shape()[0], 1u);
    ASSERT_EQ(rotated.shape()[1], 3u);
    ASSERT_EQ(rotated.shape()[2], 4u);

    const auto rotated_acc = rotated.accessor<float, 3>();
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 0), 3.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 1), 6.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 2), 9.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 0, 3), 12.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 2, 0), 1.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 2, 1), 4.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 2, 2), 7.0f);
    EXPECT_FLOAT_EQ(rotated_acc(0, 2, 3), 10.0f);

    const auto mask = Tensor::from_vector(
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         10.0f, 11.0f, 12.0f},
        TensorShape({4, 3}),
        Device::CPU);
    const auto rotated_mask =
        camera.normalize_loaded_mask_orientation(mask).cpu();
    const auto mask_acc = rotated_mask.accessor<float, 2>();
    ASSERT_EQ(rotated_mask.shape()[0], 3u);
    ASSERT_EQ(rotated_mask.shape()[1], 4u);
    EXPECT_FLOAT_EQ(mask_acc(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(mask_acc(0, 1), 6.0f);
    EXPECT_FLOAT_EQ(mask_acc(0, 2), 9.0f);
    EXPECT_FLOAT_EQ(mask_acc(0, 3), 12.0f);
    EXPECT_FLOAT_EQ(mask_acc(2, 0), 1.0f);
    EXPECT_FLOAT_EQ(mask_acc(2, 1), 4.0f);
    EXPECT_FLOAT_EQ(mask_acc(2, 2), 7.0f);
    EXPECT_FLOAT_EQ(mask_acc(2, 3), 10.0f);
}

// ====================== Per-model undistortion tests ======================

TEST(UndistortPinhole, ZeroDistortion_Noop) {
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor(), Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    EXPECT_NEAR(params.dst_width, TEST_W, TEST_W / 4);
    EXPECT_NEAR(params.dst_height, TEST_H, TEST_H / 4);
}

TEST(UndistortPinhole, SimpleRadial) {
    // COLMAP model 2: SIMPLE_RADIAL — 1 radial coeff
    auto radial = Tensor::from_vector({-0.08f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortPinhole, Radial) {
    // COLMAP model 3: RADIAL — 2 radial coeffs (k1, k2)
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, OpenCV) {
    // COLMAP model 4: OPENCV — k1,k2 + p1,p2
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortPinhole, FullOpenCV) {
    // COLMAP model 6: FULL_OPENCV — k1,k2,k3 (we cap at 3 radial) + p1,p2
    auto radial = Tensor::from_vector({-0.15f, 0.03f, -0.005f}, TensorShape({3}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.002f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, StrongBarrelDistortion) {
    auto radial = Tensor::from_vector({-0.3f, 0.1f, -0.02f}, TensorShape({3}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, StrongPincushionDistortion) {
    auto radial = Tensor::from_vector({0.3f, -0.1f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== Fisheye model tests ======================

TEST(UndistortFisheye, SimpleRadialFisheye) {
    // COLMAP model 8: SIMPLE_RADIAL_FISHEYE — 1 coeff
    auto radial = Tensor::from_vector({0.05f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortFisheye, RadialFisheye) {
    // COLMAP model 9: RADIAL_FISHEYE — 2 coeffs
    auto radial = Tensor::from_vector({0.05f, -0.01f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortFisheye, OpenCVFisheye) {
    // COLMAP model 5: OPENCV_FISHEYE — 4 coeffs (k1-k4)
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.005f, -0.001f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortFisheye, StrongFisheyeDistortion) {
    auto radial = Tensor::from_vector({0.3f, -0.1f, 0.02f, -0.005f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        300.0f, 300.0f, 320.0f, 240.0f,
        640, 480,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== Thin prism fisheye tests ======================

TEST(UndistortThinPrism, FullCoefficients) {
    // COLMAP model 10: THIN_PRISM_FISHEYE
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.001f}, TensorShape({4}), Device::CPU);
    auto tangential = Tensor::from_vector({0.0005f, -0.0003f, 0.0001f, -0.0002f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortThinPrism, RadialOnlyNoTangential) {
    auto radial = Tensor::from_vector({0.05f, -0.01f, 0.002f, -0.0005f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::THIN_PRISM_FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== Rational model tests ======================

TEST(UndistortRational, FullCoefficients) {
    auto radial = Tensor::from_vector(
        {0.11f, -0.03f, 0.004f, 0.02f, -0.005f, 0.0008f, 0.01f, -0.002f},
        TensorShape({8}), Device::CPU);
    auto tangential = Tensor::from_vector(
        {0.0006f, -0.0004f, 0.75f},
        TensorShape({3}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::RATIONAL);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

// ====================== blank_pixels parameter ======================

TEST(UndistortBlankPixels, ZeroVsNonZero) {
    auto radial = Tensor::from_vector({-0.15f, 0.03f}, TensorShape({2}), Device::CPU);

    auto tight = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE, 0.0f);

    auto loose = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE, 0.5f);

    // COLMAP-style blank pixel handling keeps focal length fixed and expands
    // the output image instead of zooming the undistorted camera.
    EXPECT_FLOAT_EQ(loose.dst_fx, tight.dst_fx);
    EXPECT_FLOAT_EQ(loose.dst_fy, tight.dst_fy);
    EXPECT_GE(loose.dst_width, tight.dst_width);
    EXPECT_GE(loose.dst_height, tight.dst_height);

    // Both must produce valid results
    validate_params(tight, TEST_W, TEST_H);
    validate_params(loose, TEST_W, TEST_H);
}

TEST(UndistortColmapParity, SimpleRadialNoBlankPixelsMatchesColmap) {
    constexpr float kColmapParityTolerance = 1e-4f;
    auto radial = Tensor::from_vector({0.5f}, TensorShape({1}), Device::CPU);

    const auto params = compute_undistort_params(
        100.0f, 100.0f, 50.0f, 50.0f, 100, 100,
        radial, Tensor(), CameraModelType::PINHOLE, 0.0f);

    EXPECT_NEAR(params.dst_fx, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_fy, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cx, 42.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cy, 42.0f, kColmapParityTolerance);
    EXPECT_EQ(params.dst_width, 84);
    EXPECT_EQ(params.dst_height, 84);
}

TEST(UndistortColmapParity, SimpleRadialAllowBlankPixelsMatchesColmap) {
    constexpr float kColmapParityTolerance = 1e-4f;
    auto radial = Tensor::from_vector({0.5f}, TensorShape({1}), Device::CPU);

    const auto params = compute_undistort_params(
        100.0f, 100.0f, 50.0f, 50.0f, 100, 100,
        radial, Tensor(), CameraModelType::PINHOLE, 1.0f);

    EXPECT_NEAR(params.dst_fx, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_fy, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cx, 45.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cy, 45.0f, kColmapParityTolerance);
    EXPECT_EQ(params.dst_width, 90);
    EXPECT_EQ(params.dst_height, 90);
}

TEST(ScaleUndistortParams, PreservesPrincipalPointOffset) {
    UndistortParams params{};
    params.src_fx = 100.0f;
    params.src_fy = 120.0f;
    params.src_cx = 40.0f;
    params.src_cy = 15.0f;
    params.src_width = 80;
    params.src_height = 30;
    params.dst_fx = 100.0f;
    params.dst_fy = 120.0f;
    params.dst_cx = 33.0f;
    params.dst_cy = 12.0f;
    params.dst_width = 66;
    params.dst_height = 24;

    const auto scaled = scale_undistort_params(params, 40, 15);

    EXPECT_FLOAT_EQ(scaled.src_fx, 50.0f);
    EXPECT_FLOAT_EQ(scaled.src_fy, 60.0f);
    EXPECT_FLOAT_EQ(scaled.src_cx, 20.0f);
    EXPECT_FLOAT_EQ(scaled.src_cy, 7.5f);
    EXPECT_FLOAT_EQ(scaled.dst_fx, 50.0f);
    EXPECT_FLOAT_EQ(scaled.dst_fy, 60.0f);
    EXPECT_FLOAT_EQ(scaled.dst_cx, 16.5f);
    EXPECT_FLOAT_EQ(scaled.dst_cy, 6.0f);
    EXPECT_EQ(scaled.dst_width, 33);
    EXPECT_EQ(scaled.dst_height, 12);
}

// ====================== Mask-image consistency ======================

TEST(UndistortConsistency, MaskAndImageSameDimensions) {
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    auto img_src = Tensor::randn({3, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);
    auto mask_src = Tensor::ones({static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);

    auto img_dst = undistort_image(img_src, params, nullptr);
    auto mask_dst = undistort_mask(mask_src, params, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(img_dst.shape()[1], mask_dst.shape()[0]);
    EXPECT_EQ(img_dst.shape()[2], mask_dst.shape()[1]);
}

// ====================== Center pixel preservation ======================

TEST(UndistortCenter, CenterPixelPreserved) {
    // For radial distortion, the center of distortion should map to itself.
    // Create a white image with a single bright pixel at center.
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor::from_vector({-0.1f, 0.01f}, TensorShape({2}), Device::CPU),
        Tensor(), CameraModelType::PINHOLE);

    auto src = Tensor::zeros({1, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);
    auto src_cpu = src.cpu();
    auto acc = src_cpu.accessor<float, 3>();
    int cx = static_cast<int>(TEST_CX);
    int cy = static_cast<int>(TEST_CY);
    acc(0, cy, cx) = 1.0f;
    src = src_cpu.to(Device::CUDA);

    auto dst = undistort_image(src, params, nullptr);
    cudaDeviceSynchronize();

    // The brightest pixel in the output should be near the destination principal point
    auto dst_cpu = dst.cpu();
    auto dst_acc = dst_cpu.accessor<float, 3>();
    float max_val = 0.0f;
    int max_x = 0, max_y = 0;
    for (int y = 0; y < params.dst_height; ++y) {
        for (int x = 0; x < params.dst_width; ++x) {
            float v = dst_acc(0, y, x);
            if (v > max_val) {
                max_val = v;
                max_x = x;
                max_y = y;
            }
        }
    }
    EXPECT_NEAR(max_x, static_cast<int>(params.dst_cx), 3);
    EXPECT_NEAR(max_y, static_cast<int>(params.dst_cy), 3);
}

// ====================== Camera class integration (bicycle data) ======================

class UndistortCameraTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_path_ = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
        if (!std::filesystem::exists(base_path_ / "sparse" / "0" / "cameras.bin")) {
            GTEST_SKIP() << "Bicycle dataset not available";
        }

        auto result = lfs::io::read_colmap_cameras_and_images(base_path_, "images_4");
        ASSERT_TRUE(result.has_value()) << "Failed to load COLMAP data";
        auto& [cams, center] = *result;
        cameras_ = std::move(cams);
        ASSERT_GT(cameras_.size(), 0u);
    }

    std::filesystem::path base_path_;
    std::vector<std::shared_ptr<Camera>> cameras_;
};

TEST_F(UndistortCameraTest, BicyclePinholeNoDist) {
    // Bicycle is pure pinhole — prepare_undistortion should be a noop
    auto& cam = cameras_[0];
    EXPECT_FALSE(cam->has_distortion());
    cam->prepare_undistortion();
    EXPECT_FALSE(cam->is_undistort_prepared());
}

TEST_F(UndistortCameraTest, HasDistortionDetection) {
    auto& cam = cameras_[0];

    // Bicycle has no distortion
    EXPECT_FALSE(cam->has_distortion());

    // A camera constructed with radial params should report distortion
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({-0.1f}, TensorShape({1}), Device::CPU);
    Camera distorted_cam(R, T,
                         cam->focal_x(), cam->focal_y(),
                         cam->center_x(), cam->center_y(),
                         radial, Tensor(),
                         CameraModelType::PINHOLE,
                         "test", cam->image_path(), "",
                         cam->camera_width(), cam->camera_height(), 999);
    EXPECT_TRUE(distorted_cam.has_distortion());
}

TEST_F(UndistortCameraTest, PrepareAndQueryUndistortedIntrinsics) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);

    Camera distorted_cam(R, T,
                         cam->focal_x(), cam->focal_y(),
                         cam->center_x(), cam->center_y(),
                         radial, Tensor(),
                         CameraModelType::PINHOLE,
                         "test", cam->image_path(), "",
                         cam->camera_width(), cam->camera_height(), 998);

    distorted_cam.prepare_undistortion();
    ASSERT_TRUE(distorted_cam.is_undistort_prepared());

    auto [fx, fy, cx, cy] = distorted_cam.get_intrinsics();
    EXPECT_GT(fx, 0.0f);
    EXPECT_GT(fy, 0.0f);
    EXPECT_GT(cx, 0.0f);
    EXPECT_GT(cy, 0.0f);

    // Undistorted intrinsics should differ from original when distortion is present
    auto& p = distorted_cam.undistort_params();
    EXPECT_NE(p.dst_width, cam->camera_width());
}

TEST_F(UndistortCameraTest, FisheyeCameraModel) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({0.05f, -0.01f, 0.002f, -0.0005f}, TensorShape({4}), Device::CPU);

    Camera fisheye_cam(R, T,
                       cam->focal_x(), cam->focal_y(),
                       cam->center_x(), cam->center_y(),
                       radial, Tensor(),
                       CameraModelType::FISHEYE,
                       "test_fisheye", cam->image_path(), "",
                       cam->camera_width(), cam->camera_height(), 997);

    EXPECT_TRUE(fisheye_cam.has_distortion());
    fisheye_cam.prepare_undistortion();
    ASSERT_TRUE(fisheye_cam.is_undistort_prepared());

    auto& p = fisheye_cam.undistort_params();
    EXPECT_GT(p.dst_width, 0);
    EXPECT_GT(p.dst_height, 0);
    EXPECT_EQ(p.model_type, CameraModelType::FISHEYE);
}

TEST_F(UndistortCameraTest, NonPinholeModelDetectsDistortion) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();

    // Even with no distortion coefficients, a non-pinhole model means distortion
    Camera equirect_cam(R, T,
                        cam->focal_x(), cam->focal_y(),
                        cam->center_x(), cam->center_y(),
                        Tensor(), Tensor(),
                        CameraModelType::EQUIRECTANGULAR,
                        "test_equirect", cam->image_path(), "",
                        cam->camera_width(), cam->camera_height(), 996);

    EXPECT_TRUE(equirect_cam.has_distortion());
}

TEST(UndistortScale, ScaleUndistortParams) {
    const auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    const auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);

    constexpr int HALF_W = TEST_W / 2;
    constexpr int HALF_H = TEST_H / 2;
    const auto scaled = scale_undistort_params(params, HALF_W, HALF_H);

    EXPECT_EQ(scaled.src_width, HALF_W);
    EXPECT_EQ(scaled.src_height, HALF_H);

    const float sx = static_cast<float>(HALF_W) / static_cast<float>(params.src_width);
    const float sy = static_cast<float>(HALF_H) / static_cast<float>(params.src_height);

    EXPECT_NEAR(scaled.src_fx, params.src_fx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_fy, params.src_fy * sy, 1e-4f);
    EXPECT_NEAR(scaled.src_cx, params.src_cx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_cy, params.src_cy * sy, 1e-4f);

    EXPECT_NEAR(scaled.dst_fx, params.dst_fx * sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_fy, params.dst_fy * sy, 1e-4f);

    const int expected_dst_w = std::max(1, static_cast<int>(std::lroundf(params.dst_width * sx)));
    const int expected_dst_h = std::max(1, static_cast<int>(std::lroundf(params.dst_height * sy)));
    EXPECT_EQ(scaled.dst_width, expected_dst_w);
    EXPECT_EQ(scaled.dst_height, expected_dst_h);
    EXPECT_NEAR(scaled.dst_cx, params.dst_cx * sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_cy, params.dst_cy * sy, 1e-4f);

    run_image_undistort(scaled);
    run_mask_undistort(scaled);
}
