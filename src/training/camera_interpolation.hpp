/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace lfs::training {

    /// Quaternion representation for interpolation (w, x, y, z order)
    struct Quaternion {
        float w, x, y, z;

        Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
        Quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}

        /// Normalize the quaternion
        Quaternion normalized() const {
            const float len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len < 1e-8f) {
                return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
            }
            return Quaternion(w / len, x / len, y / len, z / len);
        }

        /// Dot product with another quaternion
        float dot(const Quaternion& other) const {
            return w * other.w + x * other.x + y * other.y + z * other.z;
        }

        /// Negate the quaternion
        Quaternion operator-() const {
            return Quaternion(-w, -x, -y, -z);
        }
    };

    /// Convert 3x3 rotation matrix to quaternion
    Quaternion rotation_matrix_to_quaternion(const lfs::core::Tensor& R);

    /// Convert quaternion to 3x3 rotation matrix tensor
    lfs::core::Tensor quaternion_to_rotation_matrix(const Quaternion& q);

    /// Spherical linear interpolation between two quaternions
    /// @param q0 Start quaternion
    /// @param q1 End quaternion
    /// @param t Interpolation factor [0, 1]
    /// @return Interpolated quaternion
    inline Quaternion slerp(const Quaternion& q0, const Quaternion& q1, float t) {
        Quaternion q0_norm = q0.normalized();
        Quaternion q1_norm = q1.normalized();

        // Compute dot product
        float dot = q0_norm.dot(q1_norm);

        // If dot is negative, negate one quaternion to take shorter path
        if (dot < 0.0f) {
            q1_norm = -q1_norm;
            dot = -dot;
        }

        // If quaternions are very close, use linear interpolation
        if (dot > 0.9995f) {
            Quaternion result;
            result.w = q0_norm.w + t * (q1_norm.w - q0_norm.w);
            result.x = q0_norm.x + t * (q1_norm.x - q0_norm.x);
            result.y = q0_norm.y + t * (q1_norm.y - q0_norm.y);
            result.z = q0_norm.z + t * (q1_norm.z - q0_norm.z);
            return result.normalized();
        }

        // Standard SLERP formula
        const float theta_0 = std::acos(dot);
        const float theta = theta_0 * t;
        const float sin_theta = std::sin(theta);
        const float sin_theta_0 = std::sin(theta_0);

        const float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
        const float s1 = sin_theta / sin_theta_0;

        Quaternion result;
        result.w = s0 * q0_norm.w + s1 * q1_norm.w;
        result.x = s0 * q0_norm.x + s1 * q1_norm.x;
        result.y = s0 * q0_norm.y + s1 * q1_norm.y;
        result.z = s0 * q0_norm.z + s1 * q1_norm.z;
        return result.normalized();
    }

    /// Linear interpolation for position vectors
    inline std::array<float, 3> lerp_position(const std::array<float, 3>& p0,
                                              const std::array<float, 3>& p1,
                                              float t) {
        return {
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2])};
    }

    /// Catmull-Rom spline interpolation for smooth camera paths (GoPro-like)
    /// p0, p1, p2, p3 are control points; interpolates between p1 and p2
    /// t should be in [0, 1]
    inline std::array<float, 3> catmull_rom_position(
        const std::array<float, 3>& p0,
        const std::array<float, 3>& p1,
        const std::array<float, 3>& p2,
        const std::array<float, 3>& p3,
        float t) {
        const float t2 = t * t;
        const float t3 = t2 * t;

        std::array<float, 3> result;
        for (int i = 0; i < 3; ++i) {
            // Catmull-Rom basis functions for smooth C1 continuous curve
            result[i] = 0.5f * (
                (2.0f * p1[i]) +
                (-p0[i] + p2[i]) * t +
                (2.0f * p0[i] - 5.0f * p1[i] + 4.0f * p2[i] - p3[i]) * t2 +
                (-p0[i] + 3.0f * p1[i] - 3.0f * p2[i] + p3[i]) * t3
            );
        }
        return result;
    }

    /// Extract translation from T tensor [3]
    std::array<float, 3> tensor_to_position(const lfs::core::Tensor& T);

    /// Create T tensor from position
    lfs::core::Tensor position_to_tensor(const std::array<float, 3>& pos);

    /// Camera parameters for interpolation (intrinsics + extrinsics)
    struct InterpolatedCamera {
        lfs::core::Tensor R;       ///< Rotation matrix [3, 3]
        lfs::core::Tensor T;       ///< Translation [3]
        float focal_x;             ///< Focal length X
        float focal_y;             ///< Focal length Y
        float center_x;            ///< Principal point X
        float center_y;            ///< Principal point Y
        int image_width;           ///< Image width
        int image_height;          ///< Image height

        /// Create a Camera object from interpolated parameters
        std::unique_ptr<lfs::core::Camera> to_camera(int uid) const;
    };

    /// Generate interpolated camera path between keyframe cameras
    /// @param cameras Source cameras (keyframes)
    /// @param frames_between Number of frames to interpolate between each pair
    /// @param loop If true, add frames connecting last camera back to first
    /// @return Vector of interpolated camera parameters
    std::vector<InterpolatedCamera> interpolate_camera_path(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int frames_between,
        bool loop = false);

} // namespace lfs::training
