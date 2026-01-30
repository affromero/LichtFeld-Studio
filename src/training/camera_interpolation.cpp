/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_interpolation.hpp"
#include "core/logger.hpp"
#include <algorithm>

namespace lfs::training {

    Quaternion rotation_matrix_to_quaternion(const lfs::core::Tensor& R) {
        // R is [3, 3] tensor on CPU
        const float* data = R.ptr<float>();

        // Extract matrix elements (row-major: R[row][col] = data[row * 3 + col])
        const float r00 = data[0], r01 = data[1], r02 = data[2];
        const float r10 = data[3], r11 = data[4], r12 = data[5];
        const float r20 = data[6], r21 = data[7], r22 = data[8];

        const float trace = r00 + r11 + r22;
        Quaternion q;

        if (trace > 0) {
            const float s = 0.5f / std::sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (r21 - r12) * s;
            q.y = (r02 - r20) * s;
            q.z = (r10 - r01) * s;
        } else if (r00 > r11 && r00 > r22) {
            const float s = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
            q.w = (r21 - r12) / s;
            q.x = 0.25f * s;
            q.y = (r01 + r10) / s;
            q.z = (r02 + r20) / s;
        } else if (r11 > r22) {
            const float s = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
            q.w = (r02 - r20) / s;
            q.x = (r01 + r10) / s;
            q.y = 0.25f * s;
            q.z = (r12 + r21) / s;
        } else {
            const float s = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
            q.w = (r10 - r01) / s;
            q.x = (r02 + r20) / s;
            q.y = (r12 + r21) / s;
            q.z = 0.25f * s;
        }

        return q.normalized();
    }

    lfs::core::Tensor quaternion_to_rotation_matrix(const Quaternion& q) {
        const float w = q.w, x = q.x, y = q.y, z = q.z;

        // Compute rotation matrix elements
        const float r00 = 1.0f - 2.0f * (y * y + z * z);
        const float r01 = 2.0f * (x * y - z * w);
        const float r02 = 2.0f * (x * z + y * w);
        const float r10 = 2.0f * (x * y + z * w);
        const float r11 = 1.0f - 2.0f * (x * x + z * z);
        const float r12 = 2.0f * (y * z - x * w);
        const float r20 = 2.0f * (x * z - y * w);
        const float r21 = 2.0f * (y * z + x * w);
        const float r22 = 1.0f - 2.0f * (x * x + y * y);

        // Create tensor [3, 3] on CPU
        auto R = lfs::core::Tensor::empty({3, 3}, lfs::core::Device::CPU);
        float* data = R.ptr<float>();
        data[0] = r00;
        data[1] = r01;
        data[2] = r02;
        data[3] = r10;
        data[4] = r11;
        data[5] = r12;
        data[6] = r20;
        data[7] = r21;
        data[8] = r22;

        return R;
    }

    std::array<float, 3> tensor_to_position(const lfs::core::Tensor& T) {
        const float* data = T.ptr<float>();
        return {data[0], data[1], data[2]};
    }

    lfs::core::Tensor position_to_tensor(const std::array<float, 3>& pos) {
        auto T = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU);
        float* data = T.ptr<float>();
        data[0] = pos[0];
        data[1] = pos[1];
        data[2] = pos[2];
        return T;
    }

    std::unique_ptr<lfs::core::Camera> InterpolatedCamera::to_camera(int uid) const {
        // Create empty distortion tensors
        auto radial = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
        auto tangential = lfs::core::Tensor::zeros({2}, lfs::core::Device::CPU);

        return std::make_unique<lfs::core::Camera>(
            R.to(lfs::core::Device::CUDA),
            T.to(lfs::core::Device::CUDA),
            focal_x, focal_y,
            center_x, center_y,
            radial, tangential,
            lfs::core::CameraModelType::PINHOLE,
            "interpolated_" + std::to_string(uid),
            std::filesystem::path{}, // No image path
            std::filesystem::path{}, // No mask path
            image_width, image_height,
            uid,
            0);
    }

    std::vector<InterpolatedCamera> interpolate_camera_path(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int frames_between,
        bool loop) {

        if (cameras.empty()) {
            LOG_WARN("No cameras provided for interpolation");
            return {};
        }

        if (cameras.size() == 1) {
            // Single camera - just return its parameters
            const auto& cam = cameras[0];
            InterpolatedCamera interp;
            interp.R = cam->R().to(lfs::core::Device::CPU);
            interp.T = cam->T().to(lfs::core::Device::CPU);
            std::tie(interp.focal_x, interp.focal_y, interp.center_x, interp.center_y) =
                cam->get_intrinsics();
            interp.image_width = cam->camera_width();
            interp.image_height = cam->camera_height();
            return {interp};
        }

        // Pre-extract all camera data for Catmull-Rom spline interpolation
        // IMPORTANT: Use cam_position() for actual camera world position, NOT T()!
        // T() is the view matrix translation, not the camera position.
        // Camera position = -R^T * T = cam_position()
        std::vector<std::array<float, 3>> positions;
        std::vector<Quaternion> quaternions;
        std::vector<std::array<float, 4>> intrinsics; // fx, fy, cx, cy

        for (const auto& cam : cameras) {
            // Use cam_position() to get actual camera world position
            positions.push_back(tensor_to_position(cam->cam_position().to(lfs::core::Device::CPU)));
            quaternions.push_back(rotation_matrix_to_quaternion(cam->R().to(lfs::core::Device::CPU)));
            float fx, fy, cx, cy;
            std::tie(fx, fy, cx, cy) = cam->get_intrinsics();
            intrinsics.push_back({fx, fy, cx, cy});
        }

        const int width = cameras[0]->camera_width();
        const int height = cameras[0]->camera_height();

        std::vector<InterpolatedCamera> result;
        const size_t n = cameras.size();
        const size_t num_segments = loop ? n : n - 1;

        for (size_t seg = 0; seg < num_segments; ++seg) {
            // For Catmull-Rom, we need 4 control points: p0, p1, p2, p3
            // We interpolate between p1 and p2
            size_t i0, i1, i2, i3;
            if (loop) {
                i0 = (seg + n - 1) % n;
                i1 = seg;
                i2 = (seg + 1) % n;
                i3 = (seg + 2) % n;
            } else {
                // Clamp at boundaries for non-looping paths
                i1 = seg;
                i2 = seg + 1;
                i0 = (seg > 0) ? seg - 1 : 0;
                i3 = (seg + 2 < n) ? seg + 2 : n - 1;
            }

            const auto& p0 = positions[i0];
            const auto& p1 = positions[i1];
            const auto& p2 = positions[i2];
            const auto& p3 = positions[i3];

            const auto& q1 = quaternions[i1];
            const auto& q2 = quaternions[i2];

            const auto& intr1 = intrinsics[i1];
            const auto& intr2 = intrinsics[i2];

            // Generate interpolated frames
            const int num_frames = frames_between + 1;
            const bool include_last = (seg == num_segments - 1) && !loop;

            for (int f = 0; f < num_frames; ++f) {
                if (f == num_frames - 1 && !include_last) {
                    continue;
                }

                const float t = static_cast<float>(f) / static_cast<float>(num_frames);

                InterpolatedCamera interp;

                // Smooth rotation using SLERP (could use SQUAD for even smoother)
                const Quaternion q_interp = slerp(q1, q2, t);
                interp.R = quaternion_to_rotation_matrix(q_interp);

                // Smooth position using Catmull-Rom spline (GoPro-like smoothness)
                // pos_interp is the actual camera world position
                const auto pos_interp = catmull_rom_position(p0, p1, p2, p3, t);

                // Convert world position to view matrix translation: T = -R * position
                // For view matrix [R|T], camera at 'position' maps to origin
                const float* r_data = interp.R.ptr<float>();
                interp.T = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU);
                float* t_data = interp.T.ptr<float>();
                // T = -R * pos = -(R.row(i).dot(pos)) for each i
                t_data[0] = -(r_data[0] * pos_interp[0] + r_data[1] * pos_interp[1] + r_data[2] * pos_interp[2]);
                t_data[1] = -(r_data[3] * pos_interp[0] + r_data[4] * pos_interp[1] + r_data[5] * pos_interp[2]);
                t_data[2] = -(r_data[6] * pos_interp[0] + r_data[7] * pos_interp[1] + r_data[8] * pos_interp[2]);

                // Linear interpolation for intrinsics
                interp.focal_x = intr1[0] + t * (intr2[0] - intr1[0]);
                interp.focal_y = intr1[1] + t * (intr2[1] - intr1[1]);
                interp.center_x = intr1[2] + t * (intr2[2] - intr1[2]);
                interp.center_y = intr1[3] + t * (intr2[3] - intr1[3]);
                interp.image_width = width;
                interp.image_height = height;

                result.push_back(std::move(interp));
            }
        }

        LOG_DEBUG("Generated {} interpolated camera frames from {} keyframes (Catmull-Rom spline)",
                  result.size(), cameras.size());

        return result;
    }

} // namespace lfs::training
