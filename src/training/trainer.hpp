/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "dataset.hpp"
#include "lfs/kernels/ssim.cuh"
#include "metrics/metrics.hpp"
#include "optimizer/scheduler.hpp"
#include "progress.hpp"
#include "strategies/istrategy.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stop_token>
#include <unordered_map>

// Forward declaration for Scene
namespace lfs::vis {
    class Scene;
}

namespace lfs::training {
    class AdamOptimizer;

    // Forward declarations for G3S loss classes
    namespace losses {
        struct DepthLoss;
        struct NormalConsistencyLoss;
        class MultiViewRegularization;
    } // namespace losses
    class Trainer {
    public:
        // Legacy constructor - takes ownership of strategy and shares datasets
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits);

        /**
         * @brief New constructor - takes Scene reference (Scene owns all data)
         *
         * Scene provides:
         * - Training model via getTrainingModel() (SplatData)
         * - Train/val cameras via getTrainCameras()/getValCameras()
         *
         * Strategy type ("mcmc" or "default") is determined by params during initialize()
         */
        Trainer(lfs::vis::Scene& scene);

        // Delete copy operations
        Trainer(const Trainer&) = delete;

        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;

        Trainer& operator=(Trainer&&) = default;

        ~Trainer();

        // Initialize trainer - must be called before training
        std::expected<void, std::string> initialize(const lfs::core::param::TrainingParameters& params);

        // Check if trainer is initialized
        bool isInitialized() const { return initialized_.load(); }

        // Main training method with stop token support
        std::expected<void, std::string> train(std::stop_token stop_token = {});

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; }

        bool is_paused() const { return is_paused_.load(); }
        bool is_running() const { return is_running_.load(); }
        bool is_training_complete() const { return training_complete_.load(); }
        bool has_stopped() const { return stop_requested_.load(); }

        // Get current training state
        int get_current_iteration() const { return current_iteration_.load(); }
        float get_current_loss() const { return current_loss_.load(); }

        // just for viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

        // Allow viewer to lock for rendering
        std::shared_mutex& getRenderMutex() const { return render_mutex_; }

        const lfs::core::param::TrainingParameters& getParams() const { return params_; }
        void setParams(const lfs::core::param::TrainingParameters& params);

        std::expected<void, std::string> save_checkpoint(int iteration);
        std::expected<int, std::string> load_checkpoint(const std::filesystem::path& checkpoint_path);
        void save_final_ply_and_checkpoint(int iteration);

        // Orderly shutdown - GPU sync, wait for async saves, release resources. Idempotent.
        void shutdown();

    private:
        // Helper for deferred event emission to prevent deadlocks
        struct DeferredEvents {
            std::vector<std::function<void()>> events;

            template <typename Event>
            void add(Event&& e) {
                events.push_back([e = std::move(e)]() { e.emit(); });
            }

            ~DeferredEvents() {
                for (auto& e : events)
                    e();
            }
        };

        // Training step result
        enum class StepResult {
            Continue,
            Stop,
            Error
        };

        // Returns the background color to use at a given iteration
        lfs::core::Tensor& background_for_step(int iter);

        // Returns the resized background image for the given camera dimensions
        // Returns empty tensor if no background image is set
        lfs::core::Tensor get_background_image_for_camera(int width, int height);

        lfs::core::Tensor get_random_background_for_camera(int width, int height, int iteration);

        // Protected method for processing a single training step
        std::expected<StepResult, std::string> train_step(
            int iter,
            lfs::core::Camera* cam,
            lfs::core::Tensor gt_image,
            RenderMode render_mode,
            std::stop_token stop_token = {});

        // Compute photometric loss AND gradient manually (no autograd)
        // Returns GPU tensor for loss (avoid sync!)
        std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> compute_photometric_loss_with_gradient(
            const lfs::core::Tensor& rendered,
            const lfs::core::Tensor& gt_image,
            const lfs::core::param::OptimizationParameters& opt_params);

        struct MaskLossResult {
            lfs::core::Tensor loss;
            lfs::core::Tensor grad_image;
            lfs::core::Tensor grad_alpha;
        };

        // Masked photometric loss with optional alpha gradient
        std::expected<MaskLossResult, std::string> compute_photometric_loss_with_mask(
            const lfs::core::Tensor& rendered,
            const lfs::core::Tensor& gt_image,
            const lfs::core::Tensor& mask,
            const lfs::core::Tensor& alpha,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Validate masks exist for all cameras when mask mode is enabled
        std::expected<void, std::string> validate_masks();

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_scale_reg_loss(
            lfs::core::SplatData& splatData,
            AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Returns GPU tensor for loss (avoid sync!)
        std::expected<lfs::core::Tensor, std::string> compute_opacity_reg_loss(
            lfs::core::SplatData& splatData,
            AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Returns GPU tensor for loss (avoid sync!)
        // Penalizes Gaussians drifting from their initial positions
        std::expected<lfs::core::Tensor, std::string> compute_position_reg_loss(
            lfs::core::SplatData& splatData,
            AdamOptimizer& optimizer,
            const lfs::core::param::OptimizationParameters& opt_params);

        // Sparsity optimization - returns GPU tensor (no CPU sync)
        std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string> compute_sparsity_loss_forward(
            const int iter, const lfs::core::SplatData& splat_data);

        std::expected<void, std::string> handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data);
        std::expected<void, std::string> apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data);

        // ============================================================================
        // G3S Geometric Regularization
        // ============================================================================

        /**
         * @brief Compute G3S geometric regularization losses
         *
         * Computes:
         *   1. G3S median depth using binary search stochastic transmittance
         *   2. Normal consistency loss between Gaussian normals and depth-derived normals
         *   3. Multi-view geometric regularization (if neighbor view available)
         *
         * Returns combined G3S loss tensor on GPU (no CPU sync)
         *
         * @param iter Current iteration
         * @param cam Reference camera
         * @param rendered_depth Expected depth from standard rasterization
         * @param rendered_rgb Rendered RGB image
         * @param tile_offsets Per-tile Gaussian ranges (from rasterization)
         * @param flatten_ids Sorted Gaussian indices per tile
         * @param n_isects Number of tile-gaussian intersections
         * @return Combined G3S loss tensor or error
         */
        std::expected<lfs::core::Tensor, std::string> compute_g3s_losses(
            int iter,
            lfs::core::Camera& cam,
            const lfs::core::Tensor& rendered_depth,
            const lfs::core::Tensor& rendered_rgb,
            const int32_t* tile_offsets,
            const int32_t* flatten_ids,
            int n_isects);

        /**
         * @brief Render G3S median depth using binary search transmittance
         *
         * @param cam Camera for rendering
         * @param init_depth Initial depth estimate from standard rasterization
         * @param tile_offsets Per-tile Gaussian ranges
         * @param flatten_ids Sorted Gaussian indices per tile
         * @param n_isects Number of intersections
         * @return (median_depth, valid_mask) tensors or error
         */
        std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> render_g3s_median_depth(
            lfs::core::Camera& cam,
            const lfs::core::Tensor& init_depth,
            const int32_t* tile_offsets,
            const int32_t* flatten_ids,
            int n_isects);

        /**
         * @brief Find a neighboring camera for multi-view regularization
         *
         * Selects a camera with similar viewpoint for geometric consistency checks.
         * Uses distance and angular similarity criteria.
         *
         * @param ref_cam Reference camera
         * @return Pointer to neighbor camera or nullptr if none suitable
         */
        lfs::core::Camera* find_neighbor_camera(const lfs::core::Camera& ref_cam);

        // Cleanup method for re-initialization
        void cleanup();

        std::expected<void, std::string> initialize_bilateral_grid();

        // Handle control requests
        void handle_control_requests(int iter, std::stop_token stop_token = {});

        void save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads = true);

        // Member variables
        lfs::vis::Scene* scene_ = nullptr;            // Non-owning pointer to Scene (new mode)
        std::shared_ptr<CameraDataset> base_dataset_; // Legacy mode only - source cameras
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        lfs::core::param::TrainingParameters params_;
        std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits_;

        lfs::core::Tensor background_{};
        lfs::core::Tensor bg_mix_buffer_;
        lfs::core::Tensor bg_image_base_{};                              // Original background image [C, H, W]
        std::unordered_map<uint64_t, lfs::core::Tensor> bg_image_cache_; // Cache of resized bg images keyed by (H << 32) | W
        lfs::core::Tensor random_bg_buffer_{};                           // Reusable buffer for random background
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_ = 0;

        // Pre-loaded mask from pipelined dataloader (used in train_step)
        lfs::core::Tensor pipelined_mask_;

        // Bilateral grid for appearance modeling (optional)
        std::unique_ptr<BilateralGrid> bilateral_grid_;

        std::unique_ptr<ISparsityOptimizer> sparsity_optimizer_;

        lfs::training::kernels::MaskedFusedL1SSIMWorkspace masked_fused_workspace_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<lfs::training::MetricsEvaluator> evaluator_;

        // Single mutex that protects the model during training
        mutable std::shared_mutex render_mutex_;

        // Mutex for initialization to ensure thread safety
        mutable std::mutex init_mutex_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};
        std::atomic<bool> ready_to_start_{false};
        std::atomic<bool> initialized_{false};
        std::atomic<bool> shutdown_complete_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};

        // Async callback system
        std::function<void()> callback_;
        std::atomic<bool> callback_busy_{false};
        cudaStream_t callback_stream_ = nullptr;

        // ============================================================================
        // Position Regularization State
        // ============================================================================

        lfs::core::Tensor init_means_;  ///< [N, 3] Initial Gaussian positions for position_reg loss

        // ============================================================================
        // G3S Geometric Regularization State
        // ============================================================================

        // Pre-allocated buffers for G3S computation (to avoid allocation churn)
        lfs::core::Tensor g3s_median_depth_;      ///< [H, W] G3S median depth output
        lfs::core::Tensor g3s_valid_mask_;        ///< [H, W] Validity mask for G3S depth
        lfs::core::Tensor g3s_depth_normals_;     ///< [H, W, 3] Depth-derived surface normals
        lfs::core::Tensor g3s_gaussian_normals_;  ///< [N, 3] Computed Gaussian normals
        size_t g3s_allocated_h_ = 0;              ///< Allocated height for G3S buffers
        size_t g3s_allocated_w_ = 0;              ///< Allocated width for G3S buffers
        size_t g3s_allocated_n_ = 0;              ///< Allocated Gaussian count

        // Loss class instances (stateful for buffer reuse)
        std::unique_ptr<losses::DepthLoss> depth_loss_;
        std::unique_ptr<losses::NormalConsistencyLoss> normal_consistency_loss_;
        std::unique_ptr<losses::MultiViewRegularization> multiview_reg_;

        // Neighbor camera cache for multi-view regularization
        std::unordered_map<int, int> neighbor_camera_cache_; ///< Maps camera UID to neighbor UID
    };
} // namespace lfs::training
