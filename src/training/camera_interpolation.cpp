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

        // frames_between <= 0: emit training poses verbatim (no interpolation).
        // Every rendered frame is an exact training camera, so every frame
        // reflects the training-time PSNR — no novel-view extrapolation goop.
        // ``source_uid`` carries the original camera uid so downstream renderers
        // can apply per-camera corrections (e.g., bilateral grid).
        if (frames_between <= 0) {
            std::vector<InterpolatedCamera> result;
            result.reserve(cameras.size());
            for (const auto& cam : cameras) {
                InterpolatedCamera interp;
                interp.R = cam->R().to(lfs::core::Device::CPU);
                interp.T = cam->T().to(lfs::core::Device::CPU);
                std::tie(interp.focal_x, interp.focal_y, interp.center_x, interp.center_y) =
                    cam->get_intrinsics();
                interp.image_width = cam->camera_width();
                interp.image_height = cam->camera_height();
                interp.source_uid = cam->uid();
                result.push_back(std::move(interp));
            }
            LOG_DEBUG("frames_between<=0: returning {} training-pose cameras verbatim",
                      result.size());
            return result;
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

    std::vector<InterpolatedCamera> generate_spiral_path(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int n_frames,
        float radius,
        float revolutions) {

        if (cameras.empty() || n_frames <= 0) {
            LOG_WARN("generate_spiral_path: need cameras and n_frames > 0");
            return {};
        }

        // Build a chronologically-ordered single-camera spine. LichtFeld
        // stores the dataset camera-major (all lefts, then fronts, then
        // rights) so a naive iteration walks the rig laterally, which
        // turns a "wobble around the trajectory" into "teleport across
        // the 3-cam rig". Filter to a single physical camera name prefix
        // and sort by the numeric suffix so we advance along the capture
        // trajectory.
        auto extract_frame_idx = [](const std::string& name) -> int {
            // Strip extension if present
            const auto dot = name.rfind('.');
            std::string stem = (dot == std::string::npos) ? name : name.substr(0, dot);
            // Find last '_' then parse trailing digits
            const auto underscore = stem.rfind('_');
            if (underscore == std::string::npos || underscore + 1 >= stem.size()) return 0;
            try { return std::stoi(stem.substr(underscore + 1)); }
            catch (...) { return 0; }
        };

        // Try prefixes in order of preference; fall back to all cameras.
        static constexpr const char* PREFERRED_PREFIXES[] = {"front_", "cam_", ""};
        std::vector<std::shared_ptr<lfs::core::Camera>> spine;
        for (const char* px : PREFERRED_PREFIXES) {
            spine.clear();
            for (const auto& c : cameras) {
                if (!c) continue;
                const std::string& nm = c->image_name();
                if (*px == '\0' || nm.rfind(px, 0) == 0) {
                    spine.push_back(c);
                }
            }
            if (!spine.empty()) break;
        }
        // Sort spine by frame index
        std::sort(spine.begin(), spine.end(),
                  [&](const std::shared_ptr<lfs::core::Camera>& a,
                      const std::shared_ptr<lfs::core::Camera>& b) {
                      return extract_frame_idx(a->image_name()) <
                             extract_frame_idx(b->image_name());
                  });
        if (spine.empty()) spine = cameras;

        LOG_INFO("Spiral: spine has {} cameras (first=\"{}\" last=\"{}\"), {} output frames",
                 spine.size(),
                 spine.front()->image_name(),
                 spine.back()->image_name(),
                 n_frames);

        std::vector<InterpolatedCamera> result;
        result.reserve(static_cast<size_t>(n_frames));

        const size_t n_cams = spine.size();
        const float two_pi = 2.0f * 3.14159265358979323846f;

        // Reference intrinsics + resolution come from the first camera so
        // every spiral frame uses a consistent image size.
        float fx0, fy0, cx0, cy0;
        std::tie(fx0, fy0, cx0, cy0) = spine[0]->get_intrinsics();
        const int width = spine[0]->camera_width();
        const int height = spine[0]->camera_height();

        for (int f = 0; f < n_frames; ++f) {
            // Fractional progress along the trajectory
            const float t = static_cast<float>(f) / static_cast<float>(std::max(1, n_frames - 1));
            const size_t base_idx = std::min(n_cams - 1,
                static_cast<size_t>(t * (n_cams - 1)));
            const auto& base = spine[base_idx];

            // Base rotation (world -> camera). Reused so frames stay facing the
            // same direction as the underlying training camera.
            const auto R_cpu = base->R().to(lfs::core::Device::CPU);
            // Base camera position in world space.
            const auto pos = tensor_to_position(
                base->cam_position().to(lfs::core::Device::CPU));

            // Radial offset in camera-local coords (right, up): circle in the
            // camera plane. The right/up axes are the first two rows of R^T.
            const float theta = two_pi * revolutions * t;
            const float dx = radius * std::cos(theta);   // right
            const float dy = radius * std::sin(theta);   // up

            const float* r = R_cpu.ptr<float>();
            // R rows are world-space axes when reading row i: [r[3i], r[3i+1], r[3i+2]]
            // Camera-right in world = R^T * [1,0,0] = column 0 of R = [r[0], r[3], r[6]]
            // Camera-up    in world = R^T * [0,-1,0] (image y down) => -column 1 of R
            const float right_w[3] = {r[0], r[3], r[6]};
            const float up_w[3] = {-r[1], -r[4], -r[7]};

            std::array<float, 3> new_pos = {
                pos[0] + dx * right_w[0] + dy * up_w[0],
                pos[1] + dx * right_w[1] + dy * up_w[1],
                pos[2] + dx * right_w[2] + dy * up_w[2],
            };

            InterpolatedCamera interp;
            interp.R = R_cpu;
            interp.T = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU);
            float* t_ptr = interp.T.ptr<float>();
            // T = -R * new_pos
            t_ptr[0] = -(r[0] * new_pos[0] + r[1] * new_pos[1] + r[2] * new_pos[2]);
            t_ptr[1] = -(r[3] * new_pos[0] + r[4] * new_pos[1] + r[5] * new_pos[2]);
            t_ptr[2] = -(r[6] * new_pos[0] + r[7] * new_pos[1] + r[8] * new_pos[2]);

            interp.focal_x = fx0;
            interp.focal_y = fy0;
            interp.center_x = cx0;
            interp.center_y = cy0;
            interp.image_width = width;
            interp.image_height = height;
            // Inherit the base training camera's uid so bilateral grid
            // applies the correct per-image color correction. Frames are
            // only a tiny offset (radius ~cm) from this pose, so the base
            // bilateral slot remains the right per-image prior and the
            // output matches what a training-time forward pass looks like.
            interp.source_uid = base->uid();
            result.push_back(std::move(interp));
        }

        LOG_DEBUG("Generated {} spiral frames from {} training cams (radius={:.3f}, revolutions={:.2f})",
                  result.size(), cameras.size(), radius, revolutions);
        return result;
    }

} // namespace lfs::training
