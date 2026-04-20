/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "camera_interpolation.hpp"
#include "components/bilateral_grid.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "dataset.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::training {

    /// Configuration for video rendering
    struct VideoRenderConfig {
        int fps = 30;               ///< Frames per second for output video
        int frames_between = 0;     ///< Interpolated frames between keyframe cameras. 0 = training poses verbatim (training-time PSNR in every frame; no novel-view goop).
        bool loop = false;          ///< Create looping video (return to first camera)
        bool mip_filter = false;    ///< Apply mip filtering during rendering
        int rotation_frames = 120;  ///< Number of frames for rotation video
        float training_video_duration = 10.0f;  ///< Target duration for training.mp4 in seconds
    };

    /// Result of video rendering
    struct VideoRenderResult {
        std::filesystem::path video_path;   ///< Path to the output video file
        size_t num_frames;                  ///< Number of frames rendered
        bool success;                       ///< Whether rendering succeeded
        std::string error_message;          ///< Error message if failed
    };

    /// Renders walkthrough videos from Gaussian splat models
    class VideoRenderer {
    public:
        /// Create a video renderer with the given configuration
        explicit VideoRenderer(const VideoRenderConfig& config);

        /// Render a walkthrough video at a validation checkpoint
        /// @param iteration Current training iteration (used for naming)
        /// @param val_dataset Validation dataset with cameras to use as keyframes
        /// @param model The Gaussian splat model to render
        /// @param background Background color tensor [3]
        /// @param output_dir Base output directory (video will be saved to output_dir/videos/)
        /// @return Result containing path to video and status
        VideoRenderResult render_validation_video(
            int iteration,
            std::shared_ptr<CameraDataset> val_dataset,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            const std::filesystem::path& output_dir);

        /// Render a walkthrough video using provided cameras
        /// @param cameras Vector of cameras to use as keyframes
        /// @param model The Gaussian splat model to render
        /// @param background Background color tensor [3]
        /// @param output_path Full path for the output video file
        /// @return Result containing status and any error message
        VideoRenderResult render_video(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            const std::filesystem::path& output_path);

        /// Render a rotation/orbit video around the scene
        /// @param iteration Current training iteration (used for naming)
        /// @param cameras Vector of training cameras to compute orbit from
        /// @param model The Gaussian splat model to render
        /// @param background Background color tensor [3]
        /// @param output_dir Base output directory (video will be saved to output_dir/videos/)
        /// @return Result containing path to video and status
        VideoRenderResult render_rotation_video(
            int iteration,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            const std::filesystem::path& output_dir);

        /// Render a spiral walkthrough: walk along the training trajectory
        /// with a helical offset. Stays in-distribution (frames remain
        /// close to training poses) so quality degrades gracefully with
        /// ``radius``; set radius ~5-10% of the inter-frame baseline.
        /// @param iteration       Current training iteration (used for naming)
        /// @param cameras         Trajectory spine (usually training cameras)
        /// @param model           Gaussian splat model
        /// @param background      Background color tensor [3]
        /// @param output_dir      Base output directory; saved to output_dir/videos/spiral_iter*.mp4
        /// @param n_frames        Output frame count
        /// @param radius          Offset magnitude in scene units
        /// @param revolutions     Number of spiral turns across the path
        VideoRenderResult render_spiral_video(
            int iteration,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            const std::filesystem::path& output_dir,
            int n_frames = 180,
            float radius = 0.01f,        // ~1 cm wobble for meter-scale scenes
            float revolutions = 3.0f);   // a few gentle laps along the trajectory

        /// Capture a training progress frame
        /// @param camera Camera to render from
        /// @param model The Gaussian splat model
        /// @param background Background color tensor
        /// @param iteration Current iteration (for overlay text)
        void capture_training_frame(
            const lfs::core::Camera& camera,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            int iteration);

        /// Write accumulated training frames to video
        /// @param output_dir Base output directory
        /// @return Result containing path to video and status
        VideoRenderResult write_training_video(const std::filesystem::path& output_dir);

        /// Clear accumulated training frames
        void clear_training_frames();

        /// Get number of accumulated training frames
        size_t num_training_frames() const { return training_frames_.size(); }

        /// Check if FFmpeg is available on the system
        static bool is_ffmpeg_available();

        /// Compute histogram entropy of an image (higher = more diverse)
        /// @param image Image tensor [H,W,C] or [C,H,W], float32 in [0,1]
        /// @return Entropy value (0 = uniform, higher = more diverse)
        static float compute_image_entropy(const lfs::core::Tensor& image);

        /// Select the camera with the most visually diverse ground truth image
        /// Uses histogram entropy to prefer cameras showing varied content over uniform walls/floors
        /// @param cameras Vector of cameras to choose from
        /// @param resize_factor Factor to resize images (for faster analysis)
        /// @param max_width Maximum width for image loading
        /// @return Index of the camera with highest GT image diversity
        static size_t select_diverse_camera(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            float resize_factor = 0.25f,
            int max_width = 512);

        /// Get the current configuration
        const VideoRenderConfig& config() const { return config_; }

        /// Update configuration
        void set_config(const VideoRenderConfig& config) { config_ = config; }

        /// Attach the trained bilateral grid so walkthrough frames rendered
        /// from training-camera poses (InterpolatedCamera::source_uid >= 0)
        /// get the same per-image color correction that the training loop
        /// applies at trainer.cpp:1390/1463. Without this, video frames
        /// understate the reported PSNR by 5-7 dB even at training poses.
        void set_bilateral_grid(BilateralGrid* bilateral_grid);

    private:
        /// Render frames along the interpolated camera path
        /// @param cameras Interpolated camera parameters
        /// @param model The Gaussian splat model
        /// @param background Background color tensor
        /// @param frame_dir Directory to save individual frames
        /// @return Number of frames rendered, or -1 on error
        int render_frames(
            const std::vector<InterpolatedCamera>& cameras,
            lfs::core::SplatData& model,
            lfs::core::Tensor& background,
            const std::filesystem::path& frame_dir);

        /// Encode frames to video using FFmpeg
        /// @param frame_dir Directory containing frame images
        /// @param output_path Output video path
        /// @param num_frames Number of frames to encode
        /// @return True if encoding succeeded
        bool encode_video(
            const std::filesystem::path& frame_dir,
            const std::filesystem::path& output_path,
            int num_frames);

        /// Clean up temporary frame files
        void cleanup_frames(const std::filesystem::path& frame_dir);

        /// Generate elliptical orbit camera path around the scene focus point
        /// @param cameras Source cameras to compute orbit from
        /// @param n_frames Number of frames in the orbit
        /// @return Vector of interpolated cameras along the ellipse path
        std::vector<InterpolatedCamera> generate_ellipse_path(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            int n_frames);

        VideoRenderConfig config_;
        BilateralGrid* bilateral_grid_ = nullptr;         ///< Non-owning; set via set_bilateral_grid() so render_frames can apply it for training-pose frames (source_uid >= 0).
        std::vector<lfs::core::Tensor> training_frames_;  ///< Accumulated training progress frames
        std::vector<int> training_iterations_;             ///< Iteration number for each training frame
    };

} // namespace lfs::training
