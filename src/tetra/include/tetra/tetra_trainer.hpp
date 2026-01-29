/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "tetra/tetra_features.hpp"
#include "tetra/tetra_mesh.hpp"
#include "tetra/tetra_optimizer.hpp"
#include "tetra/tetra_renderer.hpp"
#include "training/dataset.hpp"

#include <atomic>
#include <expected>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

// Forward declaration
namespace lfs::vis {
    class Scene;
}

namespace lfs::tetra {

/**
 * @brief Configuration for tetrahedral mesh training
 */
struct TetraTrainConfig {
    // Iteration counts
    int iterations = 30000;

    // Densification schedule
    int densify_start = 2000;
    int densify_end = 16000;
    int densify_interval = 500;
    float densify_grad_threshold = 0.0002f;

    // Mesh update schedule
    int delaunay_interval = 10;  // Recompute triangulation every N iterations
    int freeze_start = 18000;    // Freeze mesh connectivity after this iteration

    // Hash grid configuration
    HashGridConfig hash_config;

    // MLP configuration
    MLPConfig mlp_config;

    // Learning rates
    float vertices_lr = 1e-4f;  // Small learning rate for vertex positions
    float encoding_lr = 3e-3f;
    float network_lr = 1e-3f;

    // Learning rate schedule
    float lr_decay_start = 0.0f;        // Start decay at this fraction of training
    float lr_decay_end = 1.0f;          // End decay at this fraction
    float lr_final_multiplier = 0.1f;   // Final LR = initial * this

    // Loss weights
    float lambda_l1 = 0.8f;
    float lambda_ssim = 0.2f;

    // Regularization
    float lambda_tet_volume = 0.0001f;  // Penalize small tetrahedra
    float min_tet_volume = 1e-8f;       // Minimum allowed volume

    // Checkpointing
    int checkpoint_interval = 5000;
    std::filesystem::path output_path;

    // Evaluation
    bool enable_eval = true;                      // Enable evaluation during training
    int eval_interval = 5000;                     // Evaluate every N iterations
    bool save_eval_images = false;                // Save rendered images during eval

    // Video rendering
    bool render_rotating_video = true;            // Render rotating video after training
    int video_num_frames = 400;                   // Number of frames in rotating video
    int video_fps = 30;                           // Frames per second

    // Shell configuration
    float shell_expansion = 1.5f;

    // Image configuration
    int resize_factor = 2;  // Downscale images by this factor (matches radiance_meshes images_2)
    std::string images_folder = "images";  // Images folder within dataset
};

/**
 * @brief Training progress information
 */
struct TetraTrainProgress {
    int current_iteration = 0;
    int total_iterations = 0;
    float loss = 0.0f;
    float psnr = 0.0f;
    size_t num_vertices = 0;
    size_t num_tetrahedra = 0;
    float vertices_lr = 0.0f;
    float encoding_lr = 0.0f;
    bool mesh_frozen = false;
};

/**
 * @brief Evaluation metrics result structure
 *
 * Contains aggregated PSNR and SSIM metrics computed over a dataset split.
 * Per-image metrics are stored in the vectors for detailed analysis.
 */
struct EvalMetrics {
    float mean_psnr = 0.0f;
    float mean_ssim = 0.0f;
    float elapsed_time = 0.0f;
    int num_images = 0;
    int iteration = 0;

    std::vector<float> per_image_psnr;
    std::vector<float> per_image_ssim;

    [[nodiscard]] std::string to_string() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "PSNR: " << mean_psnr
           << ", SSIM: " << mean_ssim
           << ", Time: " << elapsed_time << "s"
           << ", Images: " << num_images;
        return ss.str();
    }

    static std::string to_csv_header() {
        return "iteration,psnr,ssim,time,num_images";
    }

    [[nodiscard]] std::string to_csv_row() const {
        std::stringstream ss;
        ss << iteration << ","
           << std::fixed << std::setprecision(6)
           << mean_psnr << ","
           << mean_ssim << ","
           << elapsed_time << ","
           << num_images;
        return ss.str();
    }
};

/**
 * @brief Trainer for tetrahedral mesh neural rendering
 *
 * Implements the training loop from Radiance Meshes:
 * 1. Initialize mesh via Delaunay triangulation
 * 2. Train hash grid + MLP features
 * 3. Periodically update triangulation
 * 4. Densify mesh at vertices with high gradients
 * 5. Freeze mesh connectivity in later training
 *
 * Uses LichtFeld's existing Adam optimizer and loss functions.
 */
class TetraTrainer {
public:
    // Legacy constructor - standalone training
    explicit TetraTrainer(const TetraTrainConfig& config);

    // Scene-based constructor - integrates with visualizer
    TetraTrainer(lfs::vis::Scene& scene, const TetraTrainConfig& config);

    ~TetraTrainer();

    // Delete copy operations
    TetraTrainer(const TetraTrainer&) = delete;
    TetraTrainer& operator=(const TetraTrainer&) = delete;

    // ------------------------------
    // INITIALIZATION
    // ------------------------------

    /**
     * @brief Initialize trainer with dataset
     * @param data_path Path to COLMAP/dataset directory
     * @return Error on failure
     */
    std::expected<void, std::string> initialize(
        const std::filesystem::path& data_path);

    /**
     * @brief Initialize trainer with existing point cloud
     * @param point_cloud Initial point cloud
     * @param dataset Camera dataset
     * @return Error on failure
     */
    std::expected<void, std::string> initialize(
        const core::PointCloud& point_cloud,
        std::shared_ptr<training::CameraDataset> dataset);

    [[nodiscard]] bool is_initialized() const { return initialized_.load(); }

    // ------------------------------
    // TRAINING
    // ------------------------------

    /**
     * @brief Run training loop
     * @param stop_token Token for graceful shutdown
     * @return Error on failure
     */
    std::expected<void, std::string> train(std::stop_token stop_token = {});

    /**
     * @brief Execute single training step
     * @param iteration Current iteration number
     * @return Loss value on success
     */
    std::expected<float, std::string> train_step(int iteration);

    // ------------------------------
    // CONTROL
    // ------------------------------

    void request_pause() { pause_requested_ = true; }
    void request_resume() { pause_requested_ = false; }
    void request_stop() { stop_requested_ = true; }

    [[nodiscard]] bool is_paused() const { return is_paused_.load(); }
    [[nodiscard]] bool is_running() const { return is_running_.load(); }
    [[nodiscard]] bool is_training_complete() const { return training_complete_.load(); }

    // ------------------------------
    // STATE ACCESS
    // ------------------------------

    [[nodiscard]] TetraTrainProgress get_progress() const;
    [[nodiscard]] int current_iteration() const { return current_iteration_.load(); }
    [[nodiscard]] float current_loss() const { return current_loss_.load(); }

    [[nodiscard]] const TetraMesh& mesh() const { return *mesh_; }
    [[nodiscard]] TetraMesh& mesh() { return *mesh_; }
    [[nodiscard]] const TetraFeatures& features() const { return *features_; }
    [[nodiscard]] TetraFeatures& features() { return *features_; }
    [[nodiscard]] const TetraRenderer& renderer() const { return *renderer_; }

    // Thread-safe rendering access
    [[nodiscard]] std::shared_mutex& render_mutex() const { return render_mutex_; }

    // ------------------------------
    // CHECKPOINTING
    // ------------------------------

    /**
     * @brief Save training checkpoint
     * @param path Output path
     * @return Error on failure
     */
    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path) const;

    /**
     * @brief Load training checkpoint
     * @param path Checkpoint path
     * @return Iteration number on success
     */
    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path);

    // ------------------------------
    // EXPORT
    // ------------------------------

    /**
     * @brief Export mesh as PLY file
     * @param path Output path
     * @param extract_surface If true, export triangle mesh; else export tetrahedra
     */
    void export_mesh(const std::filesystem::path& path,
                     bool extract_surface = true) const;

    /**
     * @brief Export rendered images for all training views
     * @param output_dir Output directory
     */
    void export_renders(const std::filesystem::path& output_dir) const;

    /**
     * @brief Render a rotating video around the scene
     * @param output_path Output video path (e.g., "rotating.mp4")
     * @param num_frames Number of frames to render (default: 400)
     * @param fps Frames per second (default: 30)
     * @return Error on failure
     */
    std::expected<void, std::string> render_video(
        const std::filesystem::path& output_path,
        int num_frames = 400,
        int fps = 30) const;

    // ------------------------------
    // EVALUATION
    // ------------------------------

    /**
     * @brief Evaluate model on dataset split
     *
     * Iterates through all cameras in the dataset, renders each view,
     * computes PSNR and SSIM metrics against ground truth images.
     *
     * @param output_dir Optional directory to save rendered images and metrics.
     *                   If empty, images are not saved.
     * @param save_images Whether to save rendered images to output_dir.
     * @return EvalMetrics containing mean PSNR, SSIM, and per-image metrics.
     */
    [[nodiscard]] std::expected<EvalMetrics, std::string> evaluate(
        const std::filesystem::path& output_dir = {},
        bool save_images = true) const;

private:
    TetraTrainConfig config_;

    // Core components
    std::unique_ptr<TetraMesh> mesh_;
    std::unique_ptr<TetraFeatures> features_;
    std::unique_ptr<TetraRenderer> renderer_;
    std::unique_ptr<TetraOptimizer> optimizer_;

    // Dataset
    std::shared_ptr<training::CameraDataset> dataset_;

    // Scene reference (for visualizer integration)
    lfs::vis::Scene* scene_ = nullptr;

    // Training state
    core::Tensor background_;
    float scene_scale_ = 1.0f;
    core::Tensor scene_center_;

    // Thread synchronization
    mutable std::shared_mutex render_mutex_;
    mutable std::mutex train_mutex_;

    // Control flags
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> is_running_{false};
    std::atomic<bool> training_complete_{false};
    std::atomic<bool> initialized_{false};

    // Progress tracking
    std::atomic<int> current_iteration_{0};
    std::atomic<float> current_loss_{0.0f};

    // ------------------------------
    // GRADIENT ACCUMULATION (for densification)
    // ------------------------------

    // Accumulated gradient norms per vertex [V_interior]
    core::Tensor vertex_gradients_;

    // Number of gradient accumulations since last densification
    int gradient_count_ = 0;

    // ------------------------------
    // LOSS COMPUTATION
    // ------------------------------

    // Pre-allocated workspace for fused L1+SSIM loss (eliminates allocation churn)
    mutable lfs::training::kernels::FusedL1SSIMWorkspace fused_l1_ssim_workspace_;

    // ------------------------------
    // INTERNAL METHODS
    // ------------------------------

    /**
     * @brief Setup optimizer with parameter groups
     */
    std::expected<void, std::string> setup_optimizer();

    /**
     * @brief Update learning rates based on schedule
     */
    void update_learning_rates(int iteration);

    /**
     * @brief Compute photometric loss and gradients
     */
    std::expected<std::pair<core::Tensor, core::Tensor>, std::string>
    compute_loss(const core::Tensor& rendered,
                 const core::Tensor& gt_image);

    /**
     * @brief Densify mesh at high-gradient vertices
     */
    std::expected<void, std::string> densify(int iteration);

    /**
     * @brief Accumulate vertex gradient norms for densification
     * @param grad_vertices Gradients w.r.t vertex positions [V, 3]
     */
    void accumulate_vertex_gradients(const core::Tensor& grad_vertices);

    /**
     * @brief Reset gradient accumulation after densification
     */
    void reset_gradient_accumulation();

    /**
     * @brief Update Delaunay triangulation
     */
    std::expected<void, std::string> update_triangulation();

    /**
     * @brief Handle periodic operations (checkpoint, densify, etc.)
     */
    void handle_periodic_operations(int iteration);
};

} // namespace lfs::tetra
