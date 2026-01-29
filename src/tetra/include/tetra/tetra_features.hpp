/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

namespace lfs::tetra {

/**
 * @brief Configuration for hash grid feature encoding
 *
 * Defaults match Python radiance_meshes implementation:
 * - L=8, F=8, log2_hashmap_size=23, base_resolution=64
 */
struct HashGridConfig {
    int num_levels = 8;            // L: Number of resolution levels (Python: args.L = 8)
    int features_per_level = 8;    // F: Features per entry (Python: args.hashmap_dim = 8)
    int log2_hashmap_size = 23;    // log2(T): Hash table size per level (Python: 8,388,608 entries)
    int base_resolution = 64;      // N_min: Coarsest resolution (Python: args.base_resolution = 64)
    float per_level_scale = 2.0f;  // b: Resolution growth factor (Python: args.per_level_scale = 2)
    int max_resolution = 8192;     // N_max: Finest resolution
};

/**
 * @brief Configuration for MLP decoder
 */
struct MLPConfig {
    int input_features = 64;      // L * F from hash grid (8 levels * 8 features = 64)
    int hidden_dim = 64;          // Hidden layer dimension (Python: args.hidden_dim = 64)
    int num_hidden_layers = 2;    // Number of hidden layers
    int output_dim = 16;          // Output features (SH coefficients)
    bool use_relu = true;         // Use ReLU activation
};

/**
 * @brief Forward pass result for features
 */
struct FeatureForwardResult {
    core::Tensor rgb;        // [N, 3] RGB values
    core::Tensor density;    // [N, 1] Density values (optional)
    core::Tensor sh;         // [N, sh_dim, 3] SH coefficients (optional)
};

/**
 * @brief Cache for MLP forward pass activations needed for backward
 *
 * Stores all intermediate values required for proper gradient computation:
 * - Layer inputs (h_l): activations after ReLU from previous layer
 * - Pre-activations (z_l): values before ReLU (W @ h + b)
 * - Final MLP output (before sigmoid extraction)
 */
struct MLPForwardCache {
    std::vector<core::Tensor> layer_inputs;      // h_0, h_1, ..., h_{L-1} (L = num_layers)
    std::vector<core::Tensor> pre_activations;   // z_0, z_1, ..., z_{L-1} (before ReLU)
    core::Tensor mlp_output;                     // Final output before sigmoid
    core::Tensor hash_encoding;                  // Input to MLP (hash grid output)
    bool valid = false;
};

/**
 * @brief Backward pass inputs for features
 */
struct FeatureBackwardInputs {
    core::Tensor grad_rgb;      // [N, 3] Gradient w.r.t RGB
    core::Tensor grad_density;  // [N, 1] Gradient w.r.t density (optional)
    core::Tensor grad_sh;       // [N, sh_dim, 3] Gradient w.r.t SH (optional)
};

/**
 * @brief Backward pass result for features
 */
struct FeatureBackwardResult {
    core::Tensor grad_positions;                   // [N, 3] Gradient w.r.t positions
    core::Tensor grad_hash_params;                 // [T] Gradient w.r.t hash table entries
    core::Tensor grad_mlp_params;                  // [M] Gradient w.r.t MLP parameters (flattened)
    std::vector<core::Tensor> grad_mlp_weights;   // Gradients for each MLP weight matrix
    std::vector<core::Tensor> grad_mlp_biases;    // Gradients for each MLP bias vector
};

/**
 * @brief Hash grid + MLP feature encoding for tetrahedral meshes
 *
 * Implements instant-NGP style multi-resolution hash encoding:
 * 1. Query position is encoded at L resolution levels
 * 2. Each level uses trilinear interpolation from hash table
 * 3. Concatenated features go through MLP decoder
 *
 * The hash grid enables compact, learnable positional encoding
 * that captures high-frequency details.
 */
class TetraFeatures {
public:
    TetraFeatures() = default;
    ~TetraFeatures();

    // Delete copy operations
    TetraFeatures(const TetraFeatures&) = delete;
    TetraFeatures& operator=(const TetraFeatures&) = delete;

    // Allow move operations
    TetraFeatures(TetraFeatures&&) noexcept;
    TetraFeatures& operator=(TetraFeatures&&) noexcept;

    // ------------------------------
    // INITIALIZATION
    // ------------------------------

    /**
     * @brief Initialize feature network
     * @param hash_config Hash grid configuration
     * @param mlp_config MLP decoder configuration
     * @param scene_aabb Scene bounding box [6] (min_x, min_y, min_z, max_x, max_y, max_z)
     * @return Error on failure
     */
    std::expected<void, std::string> initialize(
        const HashGridConfig& hash_config,
        const MLPConfig& mlp_config,
        const core::Tensor& scene_aabb);

    // ------------------------------
    // FORWARD PASS
    // ------------------------------

    /**
     * @brief Compute features for query positions
     *
     * @param positions [N, 3] Query positions in world space
     * @param directions [N, 3] View directions (for view-dependent effects)
     * @return FeatureForwardResult with RGB and optional density/SH
     */
    std::expected<FeatureForwardResult, std::string> forward(
        const core::Tensor& positions,
        const core::Tensor& directions) const;

    /**
     * @brief Compute features with cached activations for backward pass
     *
     * Stores intermediate activations (pre-ReLU values) needed for proper
     * gradient computation during backward pass.
     *
     * @param positions [N, 3] Query positions in world space
     * @param directions [N, 3] View directions (for view-dependent effects)
     * @return FeatureForwardResult with RGB and optional density/SH
     */
    std::expected<FeatureForwardResult, std::string> forward_with_cache(
        const core::Tensor& positions,
        const core::Tensor& directions);

    /**
     * @brief Get the cached activations from forward_with_cache
     * @return Reference to the forward cache (valid only after forward_with_cache)
     */
    [[nodiscard]] const MLPForwardCache& get_forward_cache() const { return forward_cache_; }

    /**
     * @brief Clear the forward cache to free memory
     */
    void clear_forward_cache() { forward_cache_ = MLPForwardCache{}; }

    /**
     * @brief Compute hash grid encoding only (no MLP)
     *
     * @param positions [N, 3] Query positions
     * @return Encoded features [N, L*F]
     */
    core::Tensor encode(const core::Tensor& positions) const;

    // ------------------------------
    // BACKWARD PASS
    // ------------------------------

    /**
     * @brief Compute gradients for feature parameters
     *
     * @param positions [N, 3] Query positions
     * @param directions [N, 3] View directions
     * @param grad_inputs Gradients from downstream
     * @return FeatureBackwardResult with gradients
     */
    std::expected<FeatureBackwardResult, std::string> backward(
        const core::Tensor& positions,
        const core::Tensor& directions,
        const FeatureBackwardInputs& grad_inputs);

    // ------------------------------
    // PARAMETER ACCESS
    // ------------------------------

    /**
     * @brief Get all learnable parameters
     * @return Vector of parameter tensors
     */
    [[nodiscard]] std::vector<core::Tensor*> parameters();

    /**
     * @brief Get hash table entries
     * @return Hash table tensor [num_levels * table_size, features_per_level]
     */
    [[nodiscard]] core::Tensor& hash_table() { return hash_table_; }
    [[nodiscard]] const core::Tensor& hash_table() const { return hash_table_; }

    /**
     * @brief Get MLP weights
     */
    [[nodiscard]] std::vector<core::Tensor*> mlp_weights();

    // ------------------------------
    // CONFIGURATION
    // ------------------------------

    [[nodiscard]] const HashGridConfig& hash_config() const { return hash_config_; }
    [[nodiscard]] const MLPConfig& mlp_config() const { return mlp_config_; }
    [[nodiscard]] size_t num_parameters() const;

    // ------------------------------
    // SERIALIZATION
    // ------------------------------

    void serialize(std::ostream& os) const;
    void deserialize(std::istream& is);

private:
    HashGridConfig hash_config_;
    MLPConfig mlp_config_;

    // Scene bounds for normalization
    core::Tensor scene_aabb_;  // [6] min/max corners

    // Hash grid parameters
    core::Tensor hash_table_;  // [num_levels * table_size, features_per_level]

    // MLP parameters
    std::vector<core::Tensor> mlp_weights_;
    std::vector<core::Tensor> mlp_biases_;

    // Precomputed per-level data (CPU for reference)
    std::vector<float> resolutions_;
    std::vector<size_t> offsets_;

    // GPU tensors for CUDA kernel calls
    core::Tensor resolutions_gpu_;  // [num_levels]
    core::Tensor offsets_gpu_;      // [num_levels]

    // Forward cache for backward pass (mutable to allow caching in const forward)
    mutable MLPForwardCache forward_cache_;

    bool initialized_ = false;
};

} // namespace lfs::tetra
