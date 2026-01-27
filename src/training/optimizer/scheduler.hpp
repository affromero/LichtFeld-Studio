/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <istream>
#include <ostream>
#include <vector>

namespace lfs::training {

    class AdamOptimizer;  // Forward declaration
    enum class ParamType; // Forward declaration

    // Forward declaration for EvalMetrics
    struct EvalMetrics;

    /**
     * Simple Exponential Learning Rate Scheduler
     *
     * Multiplies the learning rate by gamma at each step:
     *   lr_new = lr_current * gamma
     *
     * Controls which learning rates to update:
     * - Empty vector (default): Updates ONLY global LR (for means in MCMC)
     * - Specific params: Updates only those parameter LRs
     * - All params: Pass all_param_types() to update everything
     *
     * Example (MCMC - only global/means LR decays):
     *   AdamOptimizer optimizer(...);
     *   ExponentialLR scheduler(optimizer, 0.99);  // Only global LR
     *
     * Example (update specific params):
     *   ExponentialLR scheduler(optimizer, 0.99, {ParamType::Means, ParamType::Sh0});
     *
     * Example (update all params):
     *   ExponentialLR scheduler(optimizer, 0.99, AdamOptimizer::all_param_types());
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class ExponentialLR {
    public:
        ExponentialLR(AdamOptimizer& optimizer, double gamma,
                      std::vector<ParamType> params_to_update = {})
            : optimizer_(optimizer),
              gamma_(gamma),
              params_to_update_(params_to_update) {
        }

        void step();

        // Serialization for checkpoints
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

        // Accessors for state
        double get_gamma() const { return gamma_; }

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        std::vector<ParamType> params_to_update_; // Empty = only global LR
    };

    /**
     * Exponential Learning Rate Scheduler with Linear Warmup
     *
     * Phase 1 (Warmup): Linearly increase LR from (initial_lr * warmup_start_factor) to initial_lr
     *   lr = initial_lr * (warmup_start_factor + (1 - warmup_start_factor) * progress)
     *   where progress = current_step / warmup_steps
     *
     * Phase 2 (Decay): Exponentially decay LR
     *   lr = initial_lr * gamma^(current_step - warmup_steps)
     *
     * Controls which learning rates to update (same as ExponentialLR):
     * - Empty vector (default): Updates ONLY global LR
     * - Specific params: Updates only those parameter LRs
     * - All params: Pass all_param_types() to update everything
     *
     * Example:
     *   AdamOptimizer optimizer(...);
     *   WarmupExponentialLR scheduler(optimizer,
     *                                  gamma=0.995,           // Exponential decay rate
     *                                  warmup_steps=100,      // 100 steps warmup
     *                                  warmup_start_factor=0.1,  // Start at 10% of initial LR
     *                                  params_to_update={});  // Only global LR
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class WarmupExponentialLR {
    public:
        WarmupExponentialLR(
            AdamOptimizer& optimizer,
            double gamma,
            int warmup_steps = 0,
            double warmup_start_factor = 1.0,
            std::vector<ParamType> params_to_update = {});

        void step();

        // Get current step count
        int get_step() const { return current_step_; }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

        // Accessors for state
        double get_gamma() const { return gamma_; }
        int get_warmup_steps() const { return warmup_steps_; }
        double get_warmup_start_factor() const { return warmup_start_factor_; }
        double get_initial_lr() const { return initial_lr_; }

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        int warmup_steps_;
        double warmup_start_factor_;
        int current_step_;
        double initial_lr_;
        std::vector<ParamType> params_to_update_; // Empty = only global LR
    };

    /**
     * Metric type for ReduceLROnPlateau scheduler
     */
    enum class PlateauMetric {
        PSNR, // Peak Signal-to-Noise Ratio (higher is better)
        SSIM  // Structural Similarity Index (higher is better)
    };

    /**
     * Mode for determining if metric improved
     */
    enum class PlateauMode {
        Max, // Higher values are better (PSNR, SSIM)
        Min  // Lower values are better (loss)
    };

    /**
     * ReduceLROnPlateau Learning Rate Scheduler
     *
     * Reduces learning rate when a metric has stopped improving.
     * Unlike step-based schedulers, this is triggered by evaluation metrics.
     *
     * The scheduler monitors a specified metric (PSNR or SSIM) and reduces
     * the learning rate by a factor when no improvement is seen for 'patience'
     * number of evaluation cycles.
     *
     * Example:
     *   AdamOptimizer optimizer(...);
     *   ReduceLROnPlateau::Config config{
     *       .metric = PlateauMetric::PSNR,
     *       .mode = PlateauMode::Max,
     *       .factor = 0.5,
     *       .patience = 3,
     *       .min_lr = 1e-7,
     *       .threshold = 0.01,
     *       .cooldown = 0
     *   };
     *   ReduceLROnPlateau scheduler(optimizer, config);
     *
     *   // After each evaluation:
     *   if (scheduler.step(eval_metrics)) {
     *       LOG_INFO("LR reduced due to plateau");
     *   }
     */
    class ReduceLROnPlateau {
    public:
        struct Config {
            PlateauMetric metric = PlateauMetric::PSNR;
            PlateauMode mode = PlateauMode::Max;
            double factor = 0.5;       // LR multiplier when plateau detected
            int patience = 3;          // Evaluations without improvement before reducing LR
            double min_lr = 1e-7;      // Floor for LR reduction
            double threshold = 0.01;   // Minimum delta to count as improvement
            int cooldown = 0;          // Evaluations to wait after LR reduction before resuming monitoring
        };

        ReduceLROnPlateau(AdamOptimizer& optimizer, const Config& config);

        /**
         * Step the scheduler with evaluation metrics.
         * @param metrics The evaluation metrics from the current evaluation
         * @return true if learning rate was reduced, false otherwise
         */
        bool step(const EvalMetrics& metrics);

        // Accessors
        double get_best_metric() const { return best_metric_; }
        int get_bad_evals() const { return bad_evals_; }
        int get_cooldown_counter() const { return cooldown_counter_; }
        const Config& get_config() const { return config_; }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

    private:
        bool is_better(double current, double best) const;
        double extract_metric(const EvalMetrics& metrics) const;

        AdamOptimizer& optimizer_;
        Config config_;
        double best_metric_;
        int bad_evals_;
        int cooldown_counter_;
        bool initialized_;
    };

} // namespace lfs::training
