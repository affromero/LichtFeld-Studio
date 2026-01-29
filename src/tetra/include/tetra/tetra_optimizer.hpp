/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "tetra/tetra_features.hpp"
#include "tetra/tetra_mesh.hpp"

#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::tetra {

/**
 * @brief Configuration for tetrahedral mesh optimizer
 *
 * Manages three parameter groups with different learning rates:
 * - Vertices: Mesh vertex positions (slow, geometric)
 * - Encoding: Hash grid parameters (fast, high-frequency)
 * - Network: MLP weights (medium, learned mapping)
 */
struct TetraOptimizerConfig {
    // ------------------------------
    // LEARNING RATES
    // ------------------------------

    float vertices_lr = 1e-4f;       ///< Initial learning rate for mesh vertex positions
    float vertices_lr_final = 1e-7f; ///< Final learning rate for vertices after decay
    float encoding_lr = 3e-3f;       ///< Initial learning rate for hash grid parameters
    float encoding_lr_final = 3e-3f; ///< Final learning rate for encoding (typically no decay)
    float network_lr = 1e-3f;        ///< Initial learning rate for MLP weights
    float network_lr_final = 1e-3f;  ///< Final learning rate for network
    float pertet_lr = 3e-3f;         ///< Learning rate for per-tet parameters (density, base_color, gradient)
    float pertet_lr_final = 3e-3f;   ///< Final learning rate for per-tet params (no decay)

    // ------------------------------
    // ADAM HYPERPARAMETERS
    // ------------------------------

    float beta1 = 0.9f;   ///< First moment decay rate
    float beta2 = 0.999f; ///< Second moment decay rate
    float eps = 1e-15f;   ///< Epsilon for numerical stability (tuned for 3DGS)

    // ------------------------------
    // SCHEDULE PARAMETERS
    // ------------------------------

    int freeze_start = 18000;      ///< Iteration to freeze mesh connectivity
    int lr_delay_steps = 0;        ///< Warmup steps (Python uses 0, no delay)
    float lr_delay_mult = 1.0f;    ///< Initial LR multiplier during warmup

    // ------------------------------
    // DENSIFICATION SPIKES
    // ------------------------------

    bool enable_lr_spikes = true; ///< Enable LR spikes during densification
    int spike_duration = 20;      ///< Duration of LR spike in iterations
    int densify_interval = 500;   ///< Interval between densification operations
    int densify_end = 15000;      ///< Last iteration for densification
    int spike_midpoint = 2000;    ///< Midpoint for spike scheduling
};

/**
 * @brief Per-parameter optimizer state
 *
 * Stores Adam optimizer state for a single parameter group.
 */
struct TetraParamState {
    core::Tensor grad;       // Gradient accumulator
    core::Tensor exp_avg;    // First moment (m)
    core::Tensor exp_avg_sq; // Second moment (v)
    int64_t step_count = 0;
    size_t capacity = 0;
    size_t size = 0;
    float current_lr = 0.0f;
};

/**
 * @brief Parameter group identifier
 */
enum class TetraParamType {
    Vertices,   // Mesh vertex positions
    Encoding,   // Hash grid parameters
    Network,    // MLP weights (combined)
    Density,    // Per-tetrahedron density [T]
    BaseColor,  // Per-tetrahedron base color [T, 3]
    Gradient    // Per-tetrahedron gradient [T, 3]
};

/**
 * @brief Custom optimizer for tetrahedral mesh training
 *
 * Implements the TetOptimizer from Radiance Meshes paper:
 * - Three parameter groups with separate learning rates
 * - Exponential LR decay with optional spikes during densification
 * - Vertex addition/removal with proper optimizer state management
 *
 * Usage:
 * @code
 * TetraOptimizer optimizer(mesh, features, config);
 * optimizer.allocate_gradients();
 *
 * for (int iter = 0; iter < max_iter; ++iter) {
 *     // Backward pass fills gradients
 *     compute_gradients(optimizer.get_grad(TetraParamType::Vertices), ...);
 *
 *     optimizer.step(iter);
 *     optimizer.zero_grad();
 * }
 * @endcode
 */
class TetraOptimizer {
public:
    /**
     * @brief Construct optimizer for tetra parameters
     * @param mesh Tetrahedral mesh (owns vertices)
     * @param features Feature network (owns encoding/MLP params)
     * @param config Optimizer configuration
     */
    TetraOptimizer(TetraMesh& mesh,
                   TetraFeatures& features,
                   const TetraOptimizerConfig& config);

    ~TetraOptimizer() = default;

    // Non-copyable
    TetraOptimizer(const TetraOptimizer&) = delete;
    TetraOptimizer& operator=(const TetraOptimizer&) = delete;

    // Movable
    TetraOptimizer(TetraOptimizer&&) noexcept = default;
    TetraOptimizer& operator=(TetraOptimizer&&) noexcept = default;

    // ------------------------------
    // CORE OPTIMIZER INTERFACE
    // ------------------------------

    /**
     * @brief Execute one optimization step
     *
     * Updates parameters using Adam with bias correction:
     *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
     *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
     *   m_hat = m_t / (1 - beta1^t)
     *   v_hat = v_t / (1 - beta2^t)
     *   param = param - lr * m_hat / (sqrt(v_hat) + eps)
     *
     * @param iteration Current training iteration
     */
    void step(int iteration);

    /**
     * @brief Zero all gradient accumulators
     */
    void zero_grad();

    // ------------------------------
    // LEARNING RATE MANAGEMENT
    // ------------------------------

    /**
     * @brief Update learning rates based on schedule
     *
     * Implements exponential decay with optional spikes:
     * - Base schedule: exponential decay from init to final
     * - Warmup: delayed start with multiplier
     * - Spikes: temporary LR increase during densification
     *
     * @param iteration Current training iteration
     */
    void update_learning_rate(int iteration);

    /**
     * @brief Get current learning rate for parameter type
     */
    [[nodiscard]] float get_lr(TetraParamType type) const;

    /**
     * @brief Set learning rate for parameter type
     */
    void set_lr(TetraParamType type, float lr);

    // ------------------------------
    // GRADIENT ACCESS
    // ------------------------------

    /**
     * @brief Allocate gradient buffers
     * @param capacity Pre-allocation capacity (0 = auto)
     */
    void allocate_gradients(size_t capacity = 0);

    /**
     * @brief Check if gradients are allocated
     */
    [[nodiscard]] bool has_gradients() const;

    /**
     * @brief Get mutable gradient tensor for parameter type
     * @param type Parameter type to get gradients for
     * @return Reference to gradient tensor
     */
    core::Tensor& get_grad(TetraParamType type);

    /**
     * @brief Get const gradient tensor for parameter type
     */
    [[nodiscard]] const core::Tensor& get_grad(TetraParamType type) const;

    // ------------------------------
    // VERTEX MANAGEMENT
    // ------------------------------

    /**
     * @brief Add new vertices to the mesh with initialized optimizer state
     *
     * Extends the vertex parameter tensor and initializes optimizer state
     * (m=0, v=0) for the new vertices. Typically called during densification.
     *
     * @param new_verts [N, 3] New vertex positions
     * @return Error on failure
     */
    std::expected<void, std::string> add_vertices(const core::Tensor& new_verts);

    /**
     * @brief Remove vertices from the mesh
     *
     * Prunes vertices based on keep mask and compacts optimizer state.
     * Called during pruning to remove low-contribution vertices.
     *
     * @param keep_mask [V] Boolean mask (true = keep, false = remove)
     * @return Error on failure
     */
    std::expected<void, std::string> remove_vertices(const core::Tensor& keep_mask);

    /**
     * @brief Reset optimizer state at specific vertex indices
     *
     * Zeros out m and v for specified vertices. Called when vertices
     * are relocated or their positions are drastically changed.
     *
     * @param indices Vertex indices to reset
     */
    void reset_state_at_indices(const std::vector<int64_t>& indices);

    // ------------------------------
    // STATE ACCESS
    // ------------------------------

    /**
     * @brief Get optimizer state for parameter type
     */
    [[nodiscard]] const TetraParamState* get_state(TetraParamType type) const;

    /**
     * @brief Get step count for parameter type
     */
    [[nodiscard]] int64_t get_step_count(TetraParamType type) const;

    /**
     * @brief Check if mesh is frozen
     */
    [[nodiscard]] bool is_frozen() const { return frozen_; }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const TetraOptimizerConfig& config() const { return config_; }

    // ------------------------------
    // SERIALIZATION
    // ------------------------------

    /**
     * @brief Serialize optimizer state to stream
     */
    void serialize(std::ostream& os) const;

    /**
     * @brief Deserialize optimizer state from stream
     */
    void deserialize(std::istream& is);

private:
    TetraOptimizerConfig config_;

    // References to owned parameters
    TetraMesh& mesh_;
    TetraFeatures& features_;

    // Per-parameter-group state
    std::unordered_map<std::string, TetraParamState> states_;

    // Cached learning rates
    float vertices_lr_ = 0.0f;
    float encoding_lr_ = 0.0f;
    float network_lr_ = 0.0f;
    float pertet_lr_ = 0.0f;

    // Scene scaling factor (for vertex LR adjustment)
    float vert_lr_multi_ = 1.0f;

    // Current iteration for spike detection
    int current_iteration_ = 0;

    // Mesh frozen flag
    bool frozen_ = false;

    // ------------------------------
    // INTERNAL HELPERS
    // ------------------------------

    /**
     * @brief Get parameter tensor reference by type
     */
    core::Tensor& get_param(TetraParamType type);

    /**
     * @brief Get parameter name string
     */
    [[nodiscard]] static std::string param_name(TetraParamType type);

    /**
     * @brief Initialize optimizer state for parameter type
     */
    void init_state(TetraParamType type);

    /**
     * @brief Execute Adam step for single parameter
     */
    void step_param(TetraParamType type, int iteration);

    /**
     * @brief Compute exponential learning rate
     *
     * lr(t) = lr_init * (lr_final / lr_init)^(t / max_steps)
     * with optional warmup delay.
     */
    [[nodiscard]] float compute_exponential_lr(
        float lr_init,
        float lr_final,
        float lr_delay_mult,
        int lr_delay_steps,
        int max_steps,
        int iteration) const;

    /**
     * @brief Apply LR spike if in densification window
     */
    [[nodiscard]] float apply_lr_spike(
        float base_lr,
        float spike_lr,
        int iteration) const;

    /**
     * @brief Extend optimizer state for new parameters
     */
    void extend_state_for_new_params(TetraParamType type, size_t n_new);

    /**
     * @brief Compact optimizer state after pruning
     */
    void compact_state(TetraParamType type, const core::Tensor& keep_mask);
};

} // namespace lfs::tetra
