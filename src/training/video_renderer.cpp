/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_renderer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <numeric>

namespace lfs::training {

    // Simple 5x7 bitmap font for iteration text overlay
    // Each character is encoded as 7 bytes (rows), each byte has 5 bits (columns)
    namespace font {
        constexpr int GLYPH_WIDTH = 5;
        constexpr int GLYPH_HEIGHT = 7;

        // Get font glyph for a character (7 bytes, LSB = leftmost pixel)
        inline const uint8_t* get_glyph(char c) {
            // clang-format off
            static const uint8_t GLYPH_0[] = {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
            static const uint8_t GLYPH_1[] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
            static const uint8_t GLYPH_2[] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
            static const uint8_t GLYPH_3[] = {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E};
            static const uint8_t GLYPH_4[] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
            static const uint8_t GLYPH_5[] = {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E};
            static const uint8_t GLYPH_6[] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E};
            static const uint8_t GLYPH_7[] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
            static const uint8_t GLYPH_8[] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
            static const uint8_t GLYPH_9[] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C};
            static const uint8_t GLYPH_SPACE[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            static const uint8_t GLYPH_i[] = {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E};
            static const uint8_t GLYPH_t[] = {0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06};
            static const uint8_t GLYPH_e[] = {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E};
            static const uint8_t GLYPH_r[] = {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10};
            // clang-format on

            switch (c) {
                case '0': return GLYPH_0;
                case '1': return GLYPH_1;
                case '2': return GLYPH_2;
                case '3': return GLYPH_3;
                case '4': return GLYPH_4;
                case '5': return GLYPH_5;
                case '6': return GLYPH_6;
                case '7': return GLYPH_7;
                case '8': return GLYPH_8;
                case '9': return GLYPH_9;
                case ' ': return GLYPH_SPACE;
                case 'i': return GLYPH_i;
                case 't': return GLYPH_t;
                case 'e': return GLYPH_e;
                case 'r': return GLYPH_r;
                default: return GLYPH_SPACE;
            }
        }

        // Draw a single character on the tensor at (x, y)
        // tensor is [H,W,C] format, float32, values in [0,1]
        // Note: Image has y=0 at top, glyph row 0 is top of character
        void draw_char(lfs::core::Tensor& tensor, char c, int x, int y,
                       float fg_r, float fg_g, float fg_b,
                       float bg_r, float bg_g, float bg_b, int scale = 1) {
            const int height = static_cast<int>(tensor.shape()[0]);
            const int width = static_cast<int>(tensor.shape()[1]);
            const int channels = static_cast<int>(tensor.shape()[2]);
            float* data = tensor.ptr<float>();

            const uint8_t* glyph = get_glyph(c);

            for (int row = 0; row < GLYPH_HEIGHT; ++row) {
                for (int col = 0; col < GLYPH_WIDTH; ++col) {
                    // Flip vertically: read glyph rows from bottom to top
                    const int flipped_row = GLYPH_HEIGHT - 1 - row;
                    const bool pixel_on = (glyph[flipped_row] >> col) & 1;

                    // Draw scaled pixel
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            const int px = x + col * scale + sx;
                            const int py = y + row * scale + sy;

                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const int idx = (py * width + px) * channels;
                                if (pixel_on) {
                                    data[idx + 0] = fg_r;
                                    data[idx + 1] = fg_g;
                                    data[idx + 2] = fg_b;
                                } else {
                                    data[idx + 0] = bg_r;
                                    data[idx + 1] = bg_g;
                                    data[idx + 2] = bg_b;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw a string on the tensor
        void draw_string(lfs::core::Tensor& tensor, const std::string& text,
                         int x, int y, int scale = 1) {
            // Draw with white foreground and semi-transparent black background
            // First pass: draw background (black)
            int cur_x = x;
            for (char c : text) {
                draw_char(tensor, c, cur_x, y, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, scale);
                cur_x += (GLYPH_WIDTH + 1) * scale;
            }

            // Draw white outline by offsetting
            const int offsets[][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1},
                                       {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
            for (const auto& off : offsets) {
                cur_x = x + off[0];
                int cur_y = y + off[1];
                for (char c : text) {
                    // Only draw the foreground pixels (not background)
                    const uint8_t* glyph = get_glyph(c);
                    const int height = static_cast<int>(tensor.shape()[0]);
                    const int width = static_cast<int>(tensor.shape()[1]);
                    const int channels = static_cast<int>(tensor.shape()[2]);
                    float* data = tensor.ptr<float>();

                    for (int row = 0; row < GLYPH_HEIGHT; ++row) {
                        for (int col = 0; col < GLYPH_WIDTH; ++col) {
                            // Flip vertically: read glyph rows from bottom to top
                            const int flipped_row = GLYPH_HEIGHT - 1 - row;
                            if ((glyph[flipped_row] >> col) & 1) {
                                for (int sy = 0; sy < scale; ++sy) {
                                    for (int sx = 0; sx < scale; ++sx) {
                                        const int px = cur_x + col * scale + sx;
                                        const int py = cur_y + row * scale + sy;
                                        if (px >= 0 && px < width && py >= 0 && py < height) {
                                            const int idx = (py * width + px) * channels;
                                            data[idx + 0] = 1.0f; // White outline
                                            data[idx + 1] = 1.0f;
                                            data[idx + 2] = 1.0f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cur_x += (GLYPH_WIDTH + 1) * scale;
                }
            }

            // Final pass: draw foreground (black text on top of white outline)
            cur_x = x;
            const int cur_y = y;  // Reset to original y position
            for (char c : text) {
                const uint8_t* glyph = get_glyph(c);
                const int height = static_cast<int>(tensor.shape()[0]);
                const int width = static_cast<int>(tensor.shape()[1]);
                const int channels = static_cast<int>(tensor.shape()[2]);
                float* data = tensor.ptr<float>();

                for (int row = 0; row < GLYPH_HEIGHT; ++row) {
                    for (int col = 0; col < GLYPH_WIDTH; ++col) {
                        // Flip vertically: read glyph rows from bottom to top
                        const int flipped_row = GLYPH_HEIGHT - 1 - row;
                        if ((glyph[flipped_row] >> col) & 1) {
                            for (int sy = 0; sy < scale; ++sy) {
                                for (int sx = 0; sx < scale; ++sx) {
                                    const int px = cur_x + col * scale + sx;
                                    const int py = cur_y + row * scale + sy;
                                    if (px >= 0 && px < width && py >= 0 && py < height) {
                                        const int idx = (py * width + px) * channels;
                                        data[idx + 0] = 0.0f; // Black foreground
                                        data[idx + 1] = 0.0f;
                                        data[idx + 2] = 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }
                cur_x += (GLYPH_WIDTH + 1) * scale;
            }
        }
    } // namespace font

    VideoRenderer::VideoRenderer(const VideoRenderConfig& config)
        : config_(config) {}

    bool VideoRenderer::is_ffmpeg_available() {
        // Try to run ffmpeg -version and check return code
        const int result = std::system("ffmpeg -version > /dev/null 2>&1");
        return result == 0;
    }

    float VideoRenderer::compute_image_entropy(const lfs::core::Tensor& image) {
        // Compute histogram entropy to measure image diversity
        // Higher entropy = more diverse pixel values = more interesting image
        // Lower entropy = uniform colors (walls, floors, sky)

        auto img = image.clone().to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32);

        // Convert to HWC if needed
        if (img.ndim() == 4)
            img = img.squeeze(0);
        if (img.ndim() == 3 && img.shape()[0] <= 4)
            img = img.permute({1, 2, 0});
        img = img.contiguous();

        const int height = static_cast<int>(img.shape()[0]);
        const int width = static_cast<int>(img.shape()[1]);
        const int channels = static_cast<int>(img.shape()[2]);
        const float* data = img.ptr<float>();
        const int total_pixels = height * width;

        // Compute grayscale histogram with 64 bins
        constexpr int NUM_BINS = 64;
        std::array<int, NUM_BINS> histogram{};

        for (int i = 0; i < total_pixels; ++i) {
            // Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
            const int idx = i * channels;
            float gray = 0.0f;
            if (channels >= 3) {
                gray = 0.299f * data[idx] + 0.587f * data[idx + 1] + 0.114f * data[idx + 2];
            } else {
                gray = data[idx];
            }
            // Clamp and quantize to bin
            const int bin = std::clamp(static_cast<int>(gray * NUM_BINS), 0, NUM_BINS - 1);
            histogram[bin]++;
        }

        // Compute entropy: -sum(p * log(p))
        float entropy = 0.0f;
        for (int bin = 0; bin < NUM_BINS; ++bin) {
            if (histogram[bin] > 0) {
                const float p = static_cast<float>(histogram[bin]) / static_cast<float>(total_pixels);
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }

    size_t VideoRenderer::select_diverse_camera(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        float resize_factor,
        int max_width) {

        if (cameras.empty()) {
            return 0;
        }
        if (cameras.size() == 1) {
            return 0;
        }

        LOG_INFO("Selecting diverse camera for training video from {} candidates...", cameras.size());

        size_t best_idx = 0;
        float best_entropy = -1.0f;

        // Sample cameras (don't need to check all for large datasets)
        const size_t max_samples = std::min(cameras.size(), size_t{20});
        const size_t step = cameras.size() / max_samples;

        for (size_t sample = 0; sample < max_samples; ++sample) {
            const size_t idx = sample * step;
            if (idx >= cameras.size()) break;

            try {
                // Load image at reduced resolution for fast analysis
                auto image = cameras[idx]->load_and_get_image(resize_factor, max_width);
                if (image.numel() == 0) continue;

                const float entropy = compute_image_entropy(image);

                LOG_DEBUG("Camera {} entropy: {:.3f}", idx, entropy);

                if (entropy > best_entropy) {
                    best_entropy = entropy;
                    best_idx = idx;
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to analyze camera {}: {}", idx, e.what());
                continue;
            }
        }

        LOG_INFO("Selected camera {} with entropy {:.3f} for training video", best_idx, best_entropy);
        return best_idx;
    }

    VideoRenderResult VideoRenderer::render_validation_video(
        int iteration,
        std::shared_ptr<CameraDataset> val_dataset,
        lfs::core::SplatData& model,
        lfs::core::Tensor& background,
        const std::filesystem::path& output_dir) {

        VideoRenderResult result;
        result.success = false;

        if (!val_dataset || val_dataset->size() == 0) {
            result.error_message = "No validation cameras available for video rendering";
            LOG_WARN("{}", result.error_message);
            return result;
        }

        // Create videos directory
        const auto video_dir = output_dir / "videos";
        std::filesystem::create_directories(video_dir);

        // Generate output path
        result.video_path = video_dir / std::format("walkthrough_iter{:06d}.mp4", iteration);

        // Get cameras from validation dataset
        const auto& cameras = val_dataset->get_cameras();

        return render_video(cameras, model, background, result.video_path);
    }

    VideoRenderResult VideoRenderer::render_video(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        lfs::core::SplatData& model,
        lfs::core::Tensor& background,
        const std::filesystem::path& output_path) {

        VideoRenderResult result;
        result.video_path = output_path;
        result.success = false;

        // Check FFmpeg availability
        if (!is_ffmpeg_available()) {
            result.error_message = "FFmpeg not found - cannot encode video";
            LOG_ERROR("{}", result.error_message);
            return result;
        }

        if (cameras.empty()) {
            result.error_message = "No cameras provided for video rendering";
            LOG_WARN("{}", result.error_message);
            return result;
        }

        LOG_INFO("Rendering walkthrough video with {} keyframe cameras, {} frames between",
                 cameras.size(), config_.frames_between);

        // Use cameras in original order (dataset order is typically sequential)
        // Catmull-Rom spline interpolation handles smooth transitions
        const auto interpolated_cameras = interpolate_camera_path(
            cameras, config_.frames_between, config_.loop);

        if (interpolated_cameras.empty()) {
            result.error_message = "Failed to generate interpolated camera path";
            LOG_ERROR("{}", result.error_message);
            return result;
        }

        LOG_INFO("Interpolated {} total frames", interpolated_cameras.size());

        // Create temporary directory for frames
        const auto frame_dir = output_path.parent_path() / "_frames_temp";
        std::filesystem::create_directories(frame_dir);

        // Render all frames
        const int num_frames = render_frames(interpolated_cameras, model, background, frame_dir);
        if (num_frames <= 0) {
            result.error_message = "Failed to render video frames";
            LOG_ERROR("{}", result.error_message);
            cleanup_frames(frame_dir);
            return result;
        }

        result.num_frames = static_cast<size_t>(num_frames);

        // Encode to video
        if (!encode_video(frame_dir, output_path, num_frames)) {
            result.error_message = "Failed to encode video with FFmpeg";
            LOG_ERROR("{}", result.error_message);
            cleanup_frames(frame_dir);
            return result;
        }

        // Clean up temporary frames
        cleanup_frames(frame_dir);

        result.success = true;
        LOG_INFO("Video saved: {}", output_path.string());
        return result;
    }

    int VideoRenderer::render_frames(
        const std::vector<InterpolatedCamera>& cameras,
        lfs::core::SplatData& model,
        lfs::core::Tensor& background,
        const std::filesystem::path& frame_dir) {

        int frame_idx = 0;

        for (const auto& interp_cam : cameras) {
            // Create camera object for rendering
            auto camera = interp_cam.to_camera(frame_idx);
            if (!camera) {
                LOG_ERROR("Failed to create camera for frame {}", frame_idx);
                return -1;
            }

            // Initialize CUDA tensors for rendering
            camera->initialize_cuda_tensors();

            // Render frame
            auto render_result = fast_rasterize_forward(
                *camera, model, background,
                0, 0, 0, 0, // No tiling
                config_.mip_filter);

            if (!render_result) {
                LOG_ERROR("Failed to render frame {}: {}", frame_idx, render_result.error());
                return -1;
            }

            const auto& [output, ctx] = *render_result;

            // Save frame to disk
            const auto frame_path = frame_dir / std::format("frame_{:06d}.png", frame_idx);
            lfs::core::save_image(frame_path, output.image);

            ++frame_idx;

            // Log progress every 30 frames
            if (frame_idx % 30 == 0) {
                LOG_DEBUG("Rendered {} / {} frames", frame_idx, cameras.size());
            }
        }

        return frame_idx;
    }

    bool VideoRenderer::encode_video(
        const std::filesystem::path& frame_dir,
        const std::filesystem::path& output_path,
        int num_frames) {

        // Build FFmpeg command
        // -y: Overwrite output file
        // -framerate: Input frame rate
        // -i: Input pattern
        // -c:v libx264: Use H.264 codec
        // -pix_fmt yuv420p: Pixel format for compatibility
        // -crf 18: Quality (lower = better, 18 is visually lossless)
        // -preset medium: Encoding speed/quality tradeoff
        const std::string cmd = std::format(
            "ffmpeg -y -framerate {} -i \"{}\" "
            "-c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium "
            "\"{}\" 2>/dev/null",
            config_.fps,
            (frame_dir / "frame_%06d.png").string(),
            output_path.string());

        LOG_DEBUG("Running FFmpeg: {}", cmd);

        const int result = std::system(cmd.c_str());
        if (result != 0) {
            LOG_ERROR("FFmpeg encoding failed with code {}", result);
            return false;
        }

        // Verify output file exists
        if (!std::filesystem::exists(output_path)) {
            LOG_ERROR("FFmpeg did not create output file: {}", output_path.string());
            return false;
        }

        const auto file_size = std::filesystem::file_size(output_path);
        LOG_INFO("Encoded {} frames to video ({:.2f} MB)",
                 num_frames, static_cast<double>(file_size) / (1024.0 * 1024.0));

        return true;
    }

    void VideoRenderer::cleanup_frames(const std::filesystem::path& frame_dir) {
        if (!std::filesystem::exists(frame_dir)) {
            return;
        }

        std::error_code ec;
        const auto removed = std::filesystem::remove_all(frame_dir, ec);
        if (ec) {
            LOG_WARN("Failed to clean up frame directory {}: {}",
                     frame_dir.string(), ec.message());
        } else {
            LOG_DEBUG("Cleaned up {} temporary frame files", removed);
        }
    }

    VideoRenderResult VideoRenderer::render_rotation_video(
        int iteration,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        lfs::core::SplatData& model,
        lfs::core::Tensor& background,
        const std::filesystem::path& output_dir) {

        VideoRenderResult result;
        result.success = false;

        if (cameras.empty()) {
            result.error_message = "No cameras provided for rotation video";
            LOG_WARN("{}", result.error_message);
            return result;
        }

        // Check FFmpeg availability
        if (!is_ffmpeg_available()) {
            result.error_message = "FFmpeg not found - cannot encode video";
            LOG_ERROR("{}", result.error_message);
            return result;
        }

        // Create videos directory
        const auto video_dir = output_dir / "videos";
        std::filesystem::create_directories(video_dir);

        // Generate output path
        result.video_path = video_dir / std::format("rotating_iter{:06d}.mp4", iteration);

        LOG_INFO("Rendering rotation video with {} frames around scene", config_.rotation_frames);

        // Generate elliptical camera path
        const auto ellipse_cameras = generate_ellipse_path(cameras, config_.rotation_frames);

        if (ellipse_cameras.empty()) {
            result.error_message = "Failed to generate ellipse camera path";
            LOG_ERROR("{}", result.error_message);
            return result;
        }

        // Create temporary directory for frames
        const auto frame_dir = result.video_path.parent_path() / "_rotation_frames_temp";
        std::filesystem::create_directories(frame_dir);

        // Render all frames
        const int num_frames = render_frames(ellipse_cameras, model, background, frame_dir);
        if (num_frames <= 0) {
            result.error_message = "Failed to render rotation video frames";
            LOG_ERROR("{}", result.error_message);
            cleanup_frames(frame_dir);
            return result;
        }

        result.num_frames = static_cast<size_t>(num_frames);

        // Encode to video
        if (!encode_video(frame_dir, result.video_path, num_frames)) {
            result.error_message = "Failed to encode rotation video with FFmpeg";
            LOG_ERROR("{}", result.error_message);
            cleanup_frames(frame_dir);
            return result;
        }

        // Clean up temporary frames
        cleanup_frames(frame_dir);

        result.success = true;
        LOG_INFO("Rotation video saved: {}", result.video_path.string());
        return result;
    }

    void VideoRenderer::capture_training_frame(
        const lfs::core::Camera& camera,
        lfs::core::SplatData& model,
        lfs::core::Tensor& background,
        int iteration) {

        // Create a new camera with the same parameters (Camera doesn't have copy ctor)
        // Get camera parameters on CPU
        const auto R = camera.R().to(lfs::core::Device::CPU);
        const auto T = camera.T().to(lfs::core::Device::CPU);
        float fx, fy, cx, cy;
        std::tie(fx, fy, cx, cy) = camera.get_intrinsics();

        // Create empty distortion tensors
        auto radial = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
        auto tangential = lfs::core::Tensor::zeros({2}, lfs::core::Device::CPU);

        // Create new camera
        auto cam = std::make_unique<lfs::core::Camera>(
            R.to(lfs::core::Device::CUDA),
            T.to(lfs::core::Device::CUDA),
            fx, fy, cx, cy,
            radial, tangential,
            lfs::core::CameraModelType::PINHOLE,
            "training_frame_" + std::to_string(iteration),
            std::filesystem::path{}, // No image path
            std::filesystem::path{}, // No mask path
            camera.camera_width(), camera.camera_height(),
            iteration,
            0);

        cam->initialize_cuda_tensors();

        // Render frame
        auto render_result = fast_rasterize_forward(
            *cam, model, background,
            0, 0, 0, 0, // No tiling
            config_.mip_filter);

        if (!render_result) {
            LOG_WARN("Failed to capture training frame at iteration {}", iteration);
            return;
        }

        const auto& [output, ctx] = *render_result;

        // Store the rendered image and iteration number
        training_frames_.push_back(output.image.clone());
        training_iterations_.push_back(iteration);

        LOG_DEBUG("Captured training frame {} at iteration {}", training_frames_.size(), iteration);
    }

    VideoRenderResult VideoRenderer::write_training_video(const std::filesystem::path& output_dir) {
        VideoRenderResult result;
        result.success = false;

        if (training_frames_.empty()) {
            result.error_message = "No training frames to write";
            LOG_WARN("{}", result.error_message);
            return result;
        }

        // Check FFmpeg availability
        if (!is_ffmpeg_available()) {
            result.error_message = "FFmpeg not found - cannot encode video";
            LOG_ERROR("{}", result.error_message);
            return result;
        }

        // Create videos directory
        const auto video_dir = output_dir / "videos";
        std::filesystem::create_directories(video_dir);

        result.video_path = video_dir / "training.mp4";

        // Calculate target frame count for fixed duration video
        // E.g., 10 seconds at 30fps = 300 frames
        const int target_frames = static_cast<int>(config_.training_video_duration * config_.fps);
        const size_t num_captured = training_frames_.size();

        LOG_INFO("Writing training progress video: {} captured frames -> {} output frames ({:.1f}s @ {}fps)",
                 num_captured, target_frames, config_.training_video_duration, config_.fps);

        // Create temporary directory for frames
        const auto frame_dir = result.video_path.parent_path() / "_training_frames_temp";
        std::filesystem::create_directories(frame_dir);

        // Calculate how many output frames each captured frame should span
        // This ensures smooth playback at the target duration
        int output_frame_idx = 0;
        for (size_t i = 0; i < num_captured; ++i) {
            // Get original frame (CHW format on CUDA)
            auto original_frame = training_frames_[i].clone();

            // Clamp to valid range and move to CPU for text drawing
            original_frame = original_frame.clamp(0.0f, 1.0f)
                            .to(lfs::core::Device::CPU)
                            .to(lfs::core::DataType::Float32);

            // Convert from [C,H,W] to [H,W,C] for text drawing
            if (original_frame.ndim() == 4)
                original_frame = original_frame.squeeze(0);

            // Explicitly handle CHW -> HWC conversion
            // CHW format: shape[0] is channels (3), shape[1] is H, shape[2] is W
            lfs::core::Tensor frame_hwc;
            if (original_frame.ndim() == 3 && original_frame.shape()[0] <= 4) {
                // This is CHW format, convert to HWC
                frame_hwc = original_frame.permute({1, 2, 0}).contiguous().clone();
            } else {
                // Already HWC format
                frame_hwc = original_frame.contiguous().clone();
            }

            // Draw iteration text: "iter XXXXX"
            const int iteration = (i < training_iterations_.size())
                                  ? training_iterations_[i] : static_cast<int>(i);
            const std::string text = "iter " + std::to_string(iteration);

            // Draw text at position (10, 10) with scale 2 for visibility
            // font::draw_string expects HWC format [H, W, C]
            font::draw_string(frame_hwc, text, 10, 10, 2);

            // Calculate how many output frames this captured frame should fill
            // Distribute frames evenly: frame i should end at position (i+1) * target_frames / num_captured
            const int end_frame = static_cast<int>((i + 1) * target_frames / num_captured);
            const int frames_for_this = std::max(1, end_frame - output_frame_idx);

            // Write this frame multiple times to fill its time slot
            // save_image handles both CHW and HWC formats (converts internally)
            for (int repeat = 0; repeat < frames_for_this && output_frame_idx < target_frames; ++repeat) {
                const auto frame_path = frame_dir / std::format("frame_{:06d}.png", output_frame_idx);
                lfs::core::save_image(frame_path, frame_hwc);
                ++output_frame_idx;
            }
        }

        result.num_frames = static_cast<size_t>(output_frame_idx);

        // Encode to video (text already drawn on frames)
        if (!encode_video(frame_dir, result.video_path, output_frame_idx)) {
            result.error_message = "Failed to encode training video with FFmpeg";
            LOG_ERROR("{}", result.error_message);
            cleanup_frames(frame_dir);
            return result;
        }

        // Clean up temporary frames
        cleanup_frames(frame_dir);

        result.success = true;
        LOG_INFO("Training video saved: {}", result.video_path.string());
        return result;
    }

    void VideoRenderer::clear_training_frames() {
        training_frames_.clear();
        training_iterations_.clear();
    }

    namespace {
        // Helper functions for vector math (avoiding Eigen dependency)
        struct Vec3 {
            float x, y, z;

            Vec3() : x(0), y(0), z(0) {}
            Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

            Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
            Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
            Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
            Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

            float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
            Vec3 cross(const Vec3& o) const {
                return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
            }
            float norm() const { return std::sqrt(x * x + y * y + z * z); }
            Vec3 normalized() const {
                float n = norm();
                return n > 1e-8f ? *this / n : Vec3(0, 0, 0);
            }
        };

        // 3x3 matrix for transformations
        struct Mat3 {
            float m[9]; // row-major: m[row*3 + col]

            Mat3() { std::fill(std::begin(m), std::end(m), 0.0f); }

            static Mat3 identity() {
                Mat3 r;
                r.m[0] = r.m[4] = r.m[8] = 1.0f;
                return r;
            }

            float& operator()(int row, int col) { return m[row * 3 + col]; }
            float operator()(int row, int col) const { return m[row * 3 + col]; }

            Vec3 row(int i) const { return {m[i*3], m[i*3+1], m[i*3+2]}; }
            Vec3 col(int i) const { return {m[i], m[3+i], m[6+i]}; }

            Vec3 operator*(const Vec3& v) const {
                return {
                    m[0]*v.x + m[1]*v.y + m[2]*v.z,
                    m[3]*v.x + m[4]*v.y + m[5]*v.z,
                    m[6]*v.x + m[7]*v.y + m[8]*v.z
                };
            }

            Mat3 operator*(const Mat3& o) const {
                Mat3 r;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        r(i, j) = m[i*3]*o.m[j] + m[i*3+1]*o.m[3+j] + m[i*3+2]*o.m[6+j];
                    }
                }
                return r;
            }

            Mat3 transpose() const {
                Mat3 r;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        r(i, j) = (*this)(j, i);
                return r;
            }

            float det() const {
                return m[0] * (m[4]*m[8] - m[5]*m[7])
                     - m[1] * (m[3]*m[8] - m[5]*m[6])
                     + m[2] * (m[3]*m[7] - m[4]*m[6]);
            }

            Mat3 inverse() const {
                float d = det();
                if (std::abs(d) < 1e-10f) return identity();
                Mat3 r;
                r.m[0] = (m[4]*m[8] - m[5]*m[7]) / d;
                r.m[1] = (m[2]*m[7] - m[1]*m[8]) / d;
                r.m[2] = (m[1]*m[5] - m[2]*m[4]) / d;
                r.m[3] = (m[5]*m[6] - m[3]*m[8]) / d;
                r.m[4] = (m[0]*m[8] - m[2]*m[6]) / d;
                r.m[5] = (m[2]*m[3] - m[0]*m[5]) / d;
                r.m[6] = (m[3]*m[7] - m[4]*m[6]) / d;
                r.m[7] = (m[1]*m[6] - m[0]*m[7]) / d;
                r.m[8] = (m[0]*m[4] - m[1]*m[3]) / d;
                return r;
            }
        };

        // Calculate focus point as the nearest point to all camera look-at rays
        // This is the least-squares solution: inv(sum(I - d*d^T)) @ sum((I - d*d^T) @ o)
        Vec3 compute_focus_point(const std::vector<Vec3>& positions,
                                  const std::vector<Vec3>& directions) {
            if (positions.empty()) return Vec3(0, 0, 0);

            // Accumulate sum of (I - d*d^T) matrices and (I - d*d^T) @ origin
            Mat3 A;  // sum of M_i
            Vec3 b(0, 0, 0);  // sum of M_i @ o_i

            for (size_t i = 0; i < positions.size(); ++i) {
                const Vec3& d = directions[i];
                const Vec3& o = positions[i];

                // M = I - d * d^T (projection matrix onto plane perpendicular to d)
                Mat3 M;
                M(0, 0) = 1.0f - d.x * d.x; M(0, 1) = -d.x * d.y;       M(0, 2) = -d.x * d.z;
                M(1, 0) = -d.y * d.x;       M(1, 1) = 1.0f - d.y * d.y; M(1, 2) = -d.y * d.z;
                M(2, 0) = -d.z * d.x;       M(2, 1) = -d.z * d.y;       M(2, 2) = 1.0f - d.z * d.z;

                // Accumulate M^T @ M (which equals M @ M since M is symmetric)
                Mat3 MtM = M * M;
                for (int j = 0; j < 9; ++j) A.m[j] += MtM.m[j];

                // Accumulate M @ o
                Vec3 Mo = M * o;
                b = b + Mo;
            }

            // Average
            const float n = static_cast<float>(positions.size());
            for (int j = 0; j < 9; ++j) A.m[j] /= n;
            b = b / n;

            // Solve A @ focus = b
            return A.inverse() * b;
        }

        // Compute PCA of positions and return rotation matrix that aligns principal
        // components with XYZ axes (largest variance along X)
        Mat3 compute_pca_rotation(const std::vector<Vec3>& positions, Vec3& centroid) {
            if (positions.empty()) {
                centroid = Vec3(0, 0, 0);
                return Mat3::identity();
            }

            // Compute centroid
            centroid = Vec3(0, 0, 0);
            for (const auto& p : positions) centroid = centroid + p;
            centroid = centroid / static_cast<float>(positions.size());

            // Compute covariance matrix C = sum((p - mean) * (p - mean)^T)
            Mat3 C;
            for (const auto& p : positions) {
                Vec3 d = p - centroid;
                C(0, 0) += d.x * d.x; C(0, 1) += d.x * d.y; C(0, 2) += d.x * d.z;
                C(1, 0) += d.y * d.x; C(1, 1) += d.y * d.y; C(1, 2) += d.y * d.z;
                C(2, 0) += d.z * d.x; C(2, 1) += d.z * d.y; C(2, 2) += d.z * d.z;
            }

            // Power iteration to find eigenvectors (simple but effective for 3x3)
            // Find first eigenvector (largest eigenvalue)
            Vec3 v1(1, 0, 0);
            for (int iter = 0; iter < 50; ++iter) {
                v1 = (C * v1).normalized();
            }

            // Deflate: C' = C - lambda1 * v1 * v1^T
            float lambda1 = (C * v1).dot(v1);
            Mat3 C1 = C;
            C1(0, 0) -= lambda1 * v1.x * v1.x; C1(0, 1) -= lambda1 * v1.x * v1.y; C1(0, 2) -= lambda1 * v1.x * v1.z;
            C1(1, 0) -= lambda1 * v1.y * v1.x; C1(1, 1) -= lambda1 * v1.y * v1.y; C1(1, 2) -= lambda1 * v1.y * v1.z;
            C1(2, 0) -= lambda1 * v1.z * v1.x; C1(2, 1) -= lambda1 * v1.z * v1.y; C1(2, 2) -= lambda1 * v1.z * v1.z;

            // Find second eigenvector
            Vec3 v2(0, 1, 0);
            if (std::abs(v2.dot(v1)) > 0.9f) v2 = Vec3(0, 0, 1);
            for (int iter = 0; iter < 50; ++iter) {
                v2 = C1 * v2;
                v2 = (v2 - v1 * v2.dot(v1)).normalized(); // Orthogonalize
            }

            // Third eigenvector is cross product
            Vec3 v3 = v1.cross(v2).normalized();

            // Build rotation matrix with eigenvectors as rows
            Mat3 R;
            R(0, 0) = v1.x; R(0, 1) = v1.y; R(0, 2) = v1.z;
            R(1, 0) = v2.x; R(1, 1) = v2.y; R(1, 2) = v2.z;
            R(2, 0) = v3.x; R(2, 1) = v3.y; R(2, 2) = v3.z;

            // Ensure proper rotation (det = 1)
            if (R.det() < 0) {
                R(2, 0) = -R(2, 0); R(2, 1) = -R(2, 1); R(2, 2) = -R(2, 2);
            }

            return R;
        }

        // Resample theta values for constant-speed motion along ellipse path
        std::vector<float> resample_const_speed(const std::vector<float>& theta,
                                                  const std::vector<Vec3>& positions,
                                                  int n_samples) {
            if (positions.size() < 2) return theta;

            // Compute cumulative arc length
            std::vector<float> arc_length(positions.size());
            arc_length[0] = 0.0f;
            for (size_t i = 1; i < positions.size(); ++i) {
                float seg_len = (positions[i] - positions[i-1]).norm();
                arc_length[i] = arc_length[i-1] + seg_len;
            }

            const float total_length = arc_length.back();
            if (total_length < 1e-8f) return theta;

            // Resample at uniform arc-length intervals
            std::vector<float> new_theta(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                float target = total_length * static_cast<float>(i) / static_cast<float>(n_samples);

                // Find interval containing target arc length
                auto it = std::lower_bound(arc_length.begin(), arc_length.end(), target);
                size_t idx = std::distance(arc_length.begin(), it);
                if (idx == 0) idx = 1;
                if (idx >= arc_length.size()) idx = arc_length.size() - 1;

                // Interpolate theta
                float t = (target - arc_length[idx-1]) / (arc_length[idx] - arc_length[idx-1] + 1e-8f);
                t = std::clamp(t, 0.0f, 1.0f);
                new_theta[i] = theta[idx-1] + t * (theta[idx] - theta[idx-1]);
            }

            return new_theta;
        }

        // Percentile helper
        float percentile(std::vector<float> values, float p) {
            if (values.empty()) return 0.0f;
            std::sort(values.begin(), values.end());
            float idx = p / 100.0f * static_cast<float>(values.size() - 1);
            size_t lo = static_cast<size_t>(idx);
            size_t hi = std::min(lo + 1, values.size() - 1);
            float t = idx - static_cast<float>(lo);
            return values[lo] * (1.0f - t) + values[hi] * t;
        }
    } // anonymous namespace

    std::vector<InterpolatedCamera> VideoRenderer::generate_ellipse_path(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int n_frames) {

        if (cameras.empty()) {
            return {};
        }

        // Extract camera positions, directions (forward vectors), and up vectors
        // IMPORTANT: cam->T() is view matrix translation, NOT camera position!
        // Use cam->cam_position() for actual camera world position
        std::vector<Vec3> positions;
        std::vector<Vec3> directions;  // Camera forward/look directions
        Vec3 avg_up(0, 0, 0);

        for (const auto& cam : cameras) {
            // Get actual camera position in world coordinates
            const auto pos = cam->cam_position().to(lfs::core::Device::CPU);
            const auto R = cam->R().to(lfs::core::Device::CPU);

            const float* pos_data = pos.ptr<float>();
            const float* r_data = R.ptr<float>();

            positions.emplace_back(pos_data[0], pos_data[1], pos_data[2]);

            // R is world-to-camera rotation (w2c), stored row-major [3,3]
            // To get camera axes in world, use R^T (columns of R):
            // Column 0 = {r[0], r[3], r[6]} = camera right in world
            // Column 1 = {r[1], r[4], r[7]} = camera up in world
            // Column 2 = {r[2], r[5], r[8]} = camera +Z in world
            // Camera looks along -Z, so forward = -column2
            Vec3 fwd(-r_data[2], -r_data[5], -r_data[8]);
            directions.push_back(fwd.normalized());

            // Up vector is column 1
            avg_up = avg_up + Vec3(r_data[1], r_data[4], r_data[7]);
        }

        avg_up = avg_up.normalized();

        // Apply PCA transformation to normalize camera positions
        Vec3 centroid;
        Mat3 pca_rot = compute_pca_rotation(positions, centroid);
        Mat3 pca_rot_inv = pca_rot.transpose();

        // Transform positions and directions to PCA space
        std::vector<Vec3> pca_positions;
        std::vector<Vec3> pca_directions;
        pca_positions.reserve(positions.size());
        pca_directions.reserve(directions.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            pca_positions.push_back(pca_rot * (positions[i] - centroid));
            pca_directions.push_back((pca_rot * directions[i]).normalized());
        }

        // Compute focus point using ray intersection in PCA space
        Vec3 focus_pca = compute_focus_point(pca_positions, pca_directions);
        LOG_DEBUG("Focus point (PCA space): ({:.3f}, {:.3f}, {:.3f})",
                  focus_pca.x, focus_pca.y, focus_pca.z);

        // Compute ellipse radii using 90th percentile of camera positions relative to focus
        std::vector<float> x_offsets, y_offsets, z_values;
        for (const auto& pos : pca_positions) {
            x_offsets.push_back(std::abs(pos.x - focus_pca.x));
            y_offsets.push_back(std::abs(pos.y - focus_pca.y));
            z_values.push_back(pos.z);
        }

        const float radius_x = percentile(x_offsets, 90.0f);
        const float radius_y = percentile(y_offsets, 90.0f);
        const float z_avg = percentile(z_values, 50.0f);  // Median z height

        LOG_DEBUG("Ellipse radii: x={:.3f}, y={:.3f}, z_height={:.3f}",
                  radius_x, radius_y, z_avg);

        // Get reference camera intrinsics
        float fx, fy, cx, cy;
        std::tie(fx, fy, cx, cy) = cameras[0]->get_intrinsics();
        const int width = cameras[0]->camera_width();
        const int height = cameras[0]->camera_height();

        // Determine up vector (use principal axis closest to average up in PCA space)
        Vec3 avg_up_pca = pca_rot * avg_up;
        Vec3 up_pca;
        const float abs_x = std::abs(avg_up_pca.x);
        const float abs_y = std::abs(avg_up_pca.y);
        const float abs_z = std::abs(avg_up_pca.z);
        if (abs_y >= abs_x && abs_y >= abs_z) {
            up_pca = Vec3(0, avg_up_pca.y > 0 ? 1.0f : -1.0f, 0);
        } else if (abs_z >= abs_x) {
            up_pca = Vec3(0, 0, avg_up_pca.z > 0 ? 1.0f : -1.0f);
        } else {
            up_pca = Vec3(avg_up_pca.x > 0 ? 1.0f : -1.0f, 0, 0);
        }

        // Generate initial theta samples and positions for constant-speed resampling
        const int oversample = n_frames + 1;
        std::vector<float> theta_init(oversample);
        std::vector<Vec3> pos_init(oversample);

        for (int i = 0; i < oversample; ++i) {
            theta_init[i] = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i)
                            / static_cast<float>(oversample);

            // Position on ellipse in PCA space
            const float t_x = std::cos(theta_init[i]) * 0.5f + 0.5f;
            const float t_y = std::sin(theta_init[i]) * 0.5f + 0.5f;

            pos_init[i] = Vec3(
                focus_pca.x + (2.0f * t_x - 1.0f) * radius_x,
                focus_pca.y + (2.0f * t_y - 1.0f) * radius_y,
                z_avg
            );
        }

        // Resample for constant-speed motion
        std::vector<float> theta_resampled = resample_const_speed(theta_init, pos_init, n_frames);

        // Generate final camera path
        std::vector<InterpolatedCamera> result;
        result.reserve(n_frames);

        for (int i = 0; i < n_frames; ++i) {
            const float theta = theta_resampled[i];

            // Position on ellipse in PCA space
            const float t_x = std::cos(theta) * 0.5f + 0.5f;
            const float t_y = std::sin(theta) * 0.5f + 0.5f;

            Vec3 pos_pca(
                focus_pca.x + (2.0f * t_x - 1.0f) * radius_x,
                focus_pca.y + (2.0f * t_y - 1.0f) * radius_y,
                z_avg
            );

            // Look direction in PCA space (towards focus point)
            Vec3 look_dir_pca = (focus_pca - pos_pca).normalized();

            // Construct view matrix in PCA space
            // Right = up × look (right-handed coordinate system)
            Vec3 right_pca = up_pca.cross(look_dir_pca).normalized();
            Vec3 actual_up_pca = look_dir_pca.cross(right_pca).normalized();

            // Transform back to world space
            Vec3 pos_world = pca_rot_inv * pos_pca + centroid;
            Vec3 right_world = pca_rot_inv * right_pca;
            Vec3 up_world = pca_rot_inv * actual_up_pca;
            Vec3 fwd_world = pca_rot_inv * look_dir_pca;

            InterpolatedCamera interp;

            // Create world-to-camera rotation tensor (row-major storage)
            // For w2c rotation, rows are camera basis vectors in world coords:
            // Row 0 = camera X (right) in world
            // Row 1 = camera Y (up) in world
            // Row 2 = camera Z (back, since camera looks along -Z) in world
            interp.R = lfs::core::Tensor::empty({3, 3}, lfs::core::Device::CPU);
            float* r_data = interp.R.ptr<float>();
            // Row 0 = right
            r_data[0] = right_world.x;  r_data[1] = right_world.y;  r_data[2] = right_world.z;
            // Row 1 = up
            r_data[3] = up_world.x;     r_data[4] = up_world.y;     r_data[5] = up_world.z;
            // Row 2 = back (-forward)
            r_data[6] = -fwd_world.x;   r_data[7] = -fwd_world.y;   r_data[8] = -fwd_world.z;

            // Create view matrix translation: T = -R * position
            // For view matrix [R|T], the camera at 'position' maps to origin
            // So R * position + T = 0, therefore T = -R * position
            interp.T = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU);
            float* t_data = interp.T.ptr<float>();
            // T = -R * pos = -(R.row(0).dot(pos), R.row(1).dot(pos), R.row(2).dot(pos))
            t_data[0] = -(r_data[0] * pos_world.x + r_data[1] * pos_world.y + r_data[2] * pos_world.z);
            t_data[1] = -(r_data[3] * pos_world.x + r_data[4] * pos_world.y + r_data[5] * pos_world.z);
            t_data[2] = -(r_data[6] * pos_world.x + r_data[7] * pos_world.y + r_data[8] * pos_world.z);

            interp.focal_x = fx;
            interp.focal_y = fy;
            interp.center_x = cx;
            interp.center_y = cy;
            interp.image_width = width;
            interp.image_height = height;

            result.push_back(std::move(interp));
        }

        LOG_DEBUG("Generated {} ellipse path cameras with PCA alignment and const-speed sampling",
                  result.size());
        return result;
    }

} // namespace lfs::training
