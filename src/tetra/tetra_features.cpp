/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_features.hpp"
#include "core/logger.hpp"

#include <cmath>
#include <random>

namespace lfs::tetra {

// CUDA kernel declarations
namespace cuda {
    void launch_hash_grid_encode(
        const float* positions,
        const float* hash_table,
        const float* aabb,
        float* output,
        const float* resolutions,
        const size_t* offsets,
        size_t N,
        int num_levels,
        int features_per_level,
        size_t table_size,
        void* stream);

    void launch_hash_grid_backward(
        const float* positions,
        const float* grad_output,
        const float* aabb,
        float* grad_hash_table,
        const float* resolutions,
        const size_t* offsets,
        size_t N,
        int num_levels,
        int features_per_level,
        size_t table_size,
        void* stream);

    void launch_mlp_forward(
        const float* input,
        const float* const* weights,
        const float* const* biases,
        float* output,
        float* intermediate,
        size_t N,
        int input_dim,
        int hidden_dim,
        int output_dim,
        int num_hidden_layers,
        bool use_relu,
        void* stream);

    void launch_mlp_backward(
        const float* grad_output,
        const float* input,
        const float* intermediate,
        const float* const* weights,
        float* grad_input,
        float** grad_weights,
        float** grad_biases,
        size_t N,
        int input_dim,
        int hidden_dim,
        int output_dim,
        int num_hidden_layers,
        bool use_relu,
        void* stream);

    void launch_mlp_forward_with_cache(
        const float* input,
        const float* const* weights,
        const float* const* biases,
        float* output,
        float** layer_inputs,
        float** pre_activations,
        size_t N,
        int input_dim,
        int hidden_dim,
        int output_dim,
        int num_hidden_layers,
        bool use_relu,
        void* stream);

    void launch_mlp_backward_full(
        const float* grad_output,
        const float* const* layer_inputs,
        const float* const* pre_activations,
        const float* const* weights,
        float* grad_input,
        float** grad_weights,
        float** grad_biases,
        size_t N,
        int input_dim,
        int hidden_dim,
        int output_dim,
        int num_hidden_layers,
        bool use_relu,
        void* stream);

    void launch_relu_backward(
        const float* grad_output,
        const float* pre_activation,
        float* grad_input,
        size_t N,
        void* stream);

    void launch_weight_grad(
        const float* input,
        const float* grad_output,
        float* grad_weight,
        int N,
        int in_features,
        int out_features,
        void* stream);

    void launch_bias_grad(
        const float* grad_output,
        float* grad_bias,
        int N,
        int out_features,
        void* stream);

    void launch_matmul_relu_rowmajor(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int N,
        int in_features,
        int out_features,
        void* stream);

    void launch_matmul_bias_rowmajor(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int N,
        int in_features,
        int out_features,
        void* stream);

    void launch_extract_rgb_sigmoid(
        const float* input,
        float* output,
        size_t N,
        int output_dim,
        void* stream);

    void launch_matmul_backward(
        const float* grad_output,
        const float* weight,
        float* grad_input,
        int N,
        int out_features,
        int in_features,
        void* stream);

    void launch_hash_grid_position_backward(
        const float* positions,
        const float* grad_output,
        const float* hash_table,
        const float* aabb,
        float* grad_positions,
        const float* resolutions,
        const size_t* offsets,
        size_t N,
        int num_levels,
        int features_per_level,
        size_t table_size,
        void* stream);
}

// ------------------------------
// LIFECYCLE
// ------------------------------

TetraFeatures::~TetraFeatures() = default;

TetraFeatures::TetraFeatures(TetraFeatures&& other) noexcept
    : hash_config_(other.hash_config_)
    , mlp_config_(other.mlp_config_)
    , scene_aabb_(std::move(other.scene_aabb_))
    , hash_table_(std::move(other.hash_table_))
    , mlp_weights_(std::move(other.mlp_weights_))
    , mlp_biases_(std::move(other.mlp_biases_))
    , resolutions_(std::move(other.resolutions_))
    , offsets_(std::move(other.offsets_))
    , initialized_(other.initialized_) {
    other.initialized_ = false;
}

TetraFeatures& TetraFeatures::operator=(TetraFeatures&& other) noexcept {
    if (this != &other) {
        hash_config_ = other.hash_config_;
        mlp_config_ = other.mlp_config_;
        scene_aabb_ = std::move(other.scene_aabb_);
        hash_table_ = std::move(other.hash_table_);
        mlp_weights_ = std::move(other.mlp_weights_);
        mlp_biases_ = std::move(other.mlp_biases_);
        resolutions_ = std::move(other.resolutions_);
        offsets_ = std::move(other.offsets_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

// ------------------------------
// INITIALIZATION
// ------------------------------

std::expected<void, std::string> TetraFeatures::initialize(
    const HashGridConfig& hash_config,
    const MLPConfig& mlp_config,
    const core::Tensor& scene_aabb) {

    if (!scene_aabb.is_valid() || scene_aabb.numel() != 6) {
        return std::unexpected("scene_aabb must be [6] tensor (min_x, min_y, min_z, max_x, max_y, max_z)");
    }

    hash_config_ = hash_config;
    mlp_config_ = mlp_config;
    scene_aabb_ = scene_aabb.to(core::Device::CUDA);

    // Compute per-level resolutions and offsets
    resolutions_.resize(hash_config.num_levels);
    offsets_.resize(hash_config.num_levels + 1);

    size_t total_entries = 0;
    float b = hash_config.per_level_scale;

    for (int l = 0; l < hash_config.num_levels; ++l) {
        // Resolution at level l: N_l = floor(N_min * b^l)
        float res = std::floor(
            static_cast<float>(hash_config.base_resolution) * std::pow(b, static_cast<float>(l)));
        res = std::min(res, static_cast<float>(hash_config.max_resolution));
        resolutions_[l] = res;

        // Number of entries at this level
        size_t table_size = static_cast<size_t>(1) << hash_config.log2_hashmap_size;
        offsets_[l] = total_entries;
        total_entries += table_size;
    }
    offsets_[hash_config.num_levels] = total_entries;

    // Create GPU tensors for CUDA kernel calls
    {
        core::Tensor res_cpu = core::Tensor::empty(
            {static_cast<size_t>(hash_config.num_levels)},
            core::Device::CPU, core::DataType::Float32);
        std::memcpy(res_cpu.ptr<float>(), resolutions_.data(),
                    resolutions_.size() * sizeof(float));
        resolutions_gpu_ = res_cpu.to(core::Device::CUDA);

        // Note: kernel expects size_t* but we use Int64 (same size on 64-bit systems)
        core::Tensor off_cpu = core::Tensor::empty(
            {static_cast<size_t>(hash_config.num_levels)},
            core::Device::CPU, core::DataType::Int64);
        std::memcpy(off_cpu.ptr<int64_t>(), offsets_.data(),
                    hash_config.num_levels * sizeof(size_t));
        offsets_gpu_ = off_cpu.to(core::Device::CUDA);
    }

    LOG_INFO("Initializing hash grid with {} levels, {} total entries",
             hash_config.num_levels, total_entries);

    // Initialize hash table with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);

    core::Tensor hash_table_cpu = core::Tensor::empty(
        {total_entries, static_cast<size_t>(hash_config.features_per_level)},
        core::Device::CPU, core::DataType::Float32);

    float* hash_data = hash_table_cpu.ptr<float>();
    for (size_t i = 0; i < total_entries * hash_config.features_per_level; ++i) {
        hash_data[i] = dist(gen);
    }

    hash_table_ = hash_table_cpu.to(core::Device::CUDA);

    // Initialize MLP weights
    // Layer structure: input -> hidden -> ... -> output
    std::normal_distribution<float> weight_dist(0.0f, 0.01f);

    int input_dim = hash_config.num_levels * hash_config.features_per_level;
    if (mlp_config.input_features != input_dim) {
        LOG_WARN("MLP input_features ({}) doesn't match hash grid output ({}), using {}",
                 mlp_config.input_features, input_dim, input_dim);
    }

    mlp_weights_.clear();
    mlp_biases_.clear();

    // Input layer -> first hidden
    {
        core::Tensor w_cpu = core::Tensor::empty(
            {static_cast<size_t>(input_dim), static_cast<size_t>(mlp_config.hidden_dim)},
            core::Device::CPU, core::DataType::Float32);
        float* w = w_cpu.ptr<float>();
        for (int i = 0; i < input_dim * mlp_config.hidden_dim; ++i) {
            w[i] = weight_dist(gen);
        }
        mlp_weights_.push_back(w_cpu.to(core::Device::CUDA));

        core::Tensor b = core::Tensor::zeros(
            {static_cast<size_t>(mlp_config.hidden_dim)}, core::Device::CUDA, core::DataType::Float32);
        mlp_biases_.push_back(std::move(b));
    }

    // Hidden layers
    for (int l = 1; l < mlp_config.num_hidden_layers; ++l) {
        core::Tensor w_cpu = core::Tensor::empty(
            {static_cast<size_t>(mlp_config.hidden_dim), static_cast<size_t>(mlp_config.hidden_dim)},
            core::Device::CPU, core::DataType::Float32);
        float* w = w_cpu.ptr<float>();
        for (int i = 0; i < mlp_config.hidden_dim * mlp_config.hidden_dim; ++i) {
            w[i] = weight_dist(gen);
        }
        mlp_weights_.push_back(w_cpu.to(core::Device::CUDA));

        core::Tensor b = core::Tensor::zeros(
            {static_cast<size_t>(mlp_config.hidden_dim)}, core::Device::CUDA, core::DataType::Float32);
        mlp_biases_.push_back(std::move(b));
    }

    // Last hidden -> output
    {
        core::Tensor w_cpu = core::Tensor::empty(
            {static_cast<size_t>(mlp_config.hidden_dim), static_cast<size_t>(mlp_config.output_dim)},
            core::Device::CPU, core::DataType::Float32);
        float* w = w_cpu.ptr<float>();
        for (int i = 0; i < mlp_config.hidden_dim * mlp_config.output_dim; ++i) {
            w[i] = weight_dist(gen);
        }
        mlp_weights_.push_back(w_cpu.to(core::Device::CUDA));

        core::Tensor b = core::Tensor::zeros(
            {static_cast<size_t>(mlp_config.output_dim)}, core::Device::CUDA, core::DataType::Float32);
        mlp_biases_.push_back(std::move(b));
    }

    initialized_ = true;

    LOG_INFO("TetraFeatures initialized: {} hash entries, {} MLP params",
             total_entries * hash_config.features_per_level, num_parameters());

    return {};
}

// ------------------------------
// FORWARD PASS
// ------------------------------

std::expected<FeatureForwardResult, std::string> TetraFeatures::forward(
    const core::Tensor& positions,
    const core::Tensor& directions) const {

    if (!initialized_) {
        return std::unexpected("TetraFeatures not initialized");
    }

    if (!positions.is_valid() || positions.ndim() != 2 || positions.shape()[1] != 3) {
        return std::unexpected("positions must be [N, 3] tensor");
    }

    const size_t N = static_cast<size_t>(positions.shape()[0]);

    // ------------------------------------------------------------
    // Step 1: Hash grid encoding (already GPU-optimized)
    // ------------------------------------------------------------
    core::Tensor encoded = encode(positions);

    // ------------------------------------------------------------
    // Step 2: MLP forward pass (GPU-optimized)
    // ------------------------------------------------------------
    // All operations stay on GPU - no CPU transfers

    int enc_dim = hash_config_.num_levels * hash_config_.features_per_level;

    // Allocate intermediate buffer for hidden layer activations (ping-pong)
    // Need space for 2 hidden layer outputs to ping-pong between
    core::Tensor intermediate = core::Tensor::zeros(
        {2 * N, static_cast<size_t>(mlp_config_.hidden_dim)},
        core::Device::CUDA, core::DataType::Float32);

    // Allocate MLP output buffer
    core::Tensor mlp_output = core::Tensor::empty(
        {N, static_cast<size_t>(mlp_config_.output_dim)},
        core::Device::CUDA, core::DataType::Float32);

    // Prepare weight and bias pointers for CUDA kernel
    std::vector<const float*> weight_ptrs;
    std::vector<const float*> bias_ptrs;
    weight_ptrs.reserve(mlp_weights_.size());
    bias_ptrs.reserve(mlp_biases_.size());

    for (const auto& w : mlp_weights_) {
        weight_ptrs.push_back(w.ptr<float>());
    }
    for (const auto& b : mlp_biases_) {
        bias_ptrs.push_back(b.ptr<float>());
    }

    // Launch fused MLP forward pass on GPU
    cuda::launch_mlp_forward(
        encoded.ptr<float>(),
        weight_ptrs.data(),
        bias_ptrs.data(),
        mlp_output.ptr<float>(),
        intermediate.ptr<float>(),
        N,
        enc_dim,
        mlp_config_.hidden_dim,
        mlp_config_.output_dim,
        mlp_config_.num_hidden_layers,
        mlp_config_.use_relu,
        nullptr);  // Default stream

    // ------------------------------------------------------------
    // Step 3: Extract RGB and apply sigmoid (GPU)
    // ------------------------------------------------------------
    core::Tensor rgb = core::Tensor::empty(
        {N, 3},
        core::Device::CUDA, core::DataType::Float32);

    cuda::launch_extract_rgb_sigmoid(
        mlp_output.ptr<float>(),
        rgb.ptr<float>(),
        N,
        mlp_config_.output_dim,
        nullptr);  // Default stream

    return FeatureForwardResult{
        std::move(rgb),
        core::Tensor(),  // No density
        core::Tensor()   // No SH
    };
}

std::expected<FeatureForwardResult, std::string> TetraFeatures::forward_with_cache(
    const core::Tensor& positions,
    const core::Tensor& directions) {

    if (!initialized_) {
        return std::unexpected("TetraFeatures not initialized");
    }

    if (!positions.is_valid() || positions.ndim() != 2 || positions.shape()[1] != 3) {
        return std::unexpected("positions must be [N, 3] tensor");
    }

    const size_t N = static_cast<size_t>(positions.shape()[0]);

    // ------------------------------------------------------------
    // Step 1: Hash grid encoding (already GPU-optimized)
    // ------------------------------------------------------------
    core::Tensor encoded = encode(positions);

    // ------------------------------------------------------------
    // Step 2: MLP forward pass with cached activations
    // ------------------------------------------------------------
    int enc_dim = hash_config_.num_levels * hash_config_.features_per_level;
    int num_layers = mlp_config_.num_hidden_layers + 1;

    // Initialize forward cache
    forward_cache_.layer_inputs.clear();
    forward_cache_.pre_activations.clear();
    forward_cache_.layer_inputs.resize(static_cast<size_t>(num_layers));
    forward_cache_.pre_activations.resize(static_cast<size_t>(num_layers));
    forward_cache_.hash_encoding = encoded.clone();

    // Allocate tensors for each layer's input and pre-activation
    // Layer 0: input_dim -> hidden_dim
    // Layers 1 to num_hidden_layers-1: hidden_dim -> hidden_dim
    // Layer num_hidden_layers: hidden_dim -> output_dim

    forward_cache_.layer_inputs[0] = encoded.clone();  // h_0 = hash encoding

    for (int l = 0; l < num_layers; ++l) {
        int out_dim = (l == num_layers - 1) ? mlp_config_.output_dim : mlp_config_.hidden_dim;

        // Pre-activation z_l = W_l @ h_l + b_l
        forward_cache_.pre_activations[static_cast<size_t>(l)] = core::Tensor::empty(
            {N, static_cast<size_t>(out_dim)},
            core::Device::CUDA, core::DataType::Float32);

        // Next layer input h_{l+1} = relu(z_l) (except last layer)
        if (l < num_layers - 1) {
            forward_cache_.layer_inputs[static_cast<size_t>(l) + 1] = core::Tensor::empty(
                {N, static_cast<size_t>(out_dim)},
                core::Device::CUDA, core::DataType::Float32);
        }
    }

    // Prepare weight and bias pointers for CUDA kernel
    std::vector<const float*> weight_ptrs;
    std::vector<const float*> bias_ptrs;
    std::vector<float*> layer_input_ptrs;
    std::vector<float*> pre_act_ptrs;

    weight_ptrs.reserve(mlp_weights_.size());
    bias_ptrs.reserve(mlp_biases_.size());
    layer_input_ptrs.reserve(static_cast<size_t>(num_layers));
    pre_act_ptrs.reserve(static_cast<size_t>(num_layers));

    for (const auto& w : mlp_weights_) {
        weight_ptrs.push_back(w.ptr<float>());
    }
    for (const auto& b : mlp_biases_) {
        bias_ptrs.push_back(b.ptr<float>());
    }
    for (auto& t : forward_cache_.layer_inputs) {
        layer_input_ptrs.push_back(t.ptr<float>());
    }
    for (auto& t : forward_cache_.pre_activations) {
        pre_act_ptrs.push_back(t.ptr<float>());
    }

    // Allocate MLP output buffer
    core::Tensor mlp_output = core::Tensor::empty(
        {N, static_cast<size_t>(mlp_config_.output_dim)},
        core::Device::CUDA, core::DataType::Float32);

    // Launch MLP forward pass with caching
    cuda::launch_mlp_forward_with_cache(
        encoded.ptr<float>(),
        weight_ptrs.data(),
        bias_ptrs.data(),
        mlp_output.ptr<float>(),
        layer_input_ptrs.data(),
        pre_act_ptrs.data(),
        N,
        enc_dim,
        mlp_config_.hidden_dim,
        mlp_config_.output_dim,
        mlp_config_.num_hidden_layers,
        mlp_config_.use_relu,
        nullptr);  // Default stream

    forward_cache_.mlp_output = mlp_output.clone();
    forward_cache_.valid = true;

    // ------------------------------------------------------------
    // Step 3: Extract RGB and apply sigmoid (GPU)
    // ------------------------------------------------------------
    core::Tensor rgb = core::Tensor::empty(
        {N, 3},
        core::Device::CUDA, core::DataType::Float32);

    cuda::launch_extract_rgb_sigmoid(
        mlp_output.ptr<float>(),
        rgb.ptr<float>(),
        N,
        mlp_config_.output_dim,
        nullptr);  // Default stream

    return FeatureForwardResult{
        std::move(rgb),
        core::Tensor(),  // No density
        core::Tensor()   // No SH
    };
}

core::Tensor TetraFeatures::encode(const core::Tensor& positions) const {
    if (!initialized_) {
        return core::Tensor();
    }

    const size_t N = static_cast<size_t>(positions.shape()[0]);
    int total_features = hash_config_.num_levels * hash_config_.features_per_level;
    size_t table_size = static_cast<size_t>(1) << hash_config_.log2_hashmap_size;

    // Ensure positions are on GPU
    core::Tensor pos_gpu = positions.to(core::Device::CUDA);

    // Allocate output on GPU
    core::Tensor output = core::Tensor::zeros(
        {N, static_cast<size_t>(total_features)},
        core::Device::CUDA, core::DataType::Float32);

    // Call CUDA kernel
    // Note: offsets stored as Int64 but kernel expects size_t* (same on 64-bit)
    cuda::launch_hash_grid_encode(
        pos_gpu.ptr<float>(),
        hash_table_.ptr<float>(),
        scene_aabb_.ptr<float>(),
        output.ptr<float>(),
        resolutions_gpu_.ptr<float>(),
        reinterpret_cast<const size_t*>(offsets_gpu_.ptr<int64_t>()),
        N,
        hash_config_.num_levels,
        hash_config_.features_per_level,
        table_size,
        nullptr);  // Default stream

    return output;
}

// ------------------------------
// BACKWARD PASS
// ------------------------------

std::expected<FeatureBackwardResult, std::string> TetraFeatures::backward(
    const core::Tensor& positions,
    const core::Tensor& directions,
    const FeatureBackwardInputs& grad_inputs) {

    if (!initialized_) {
        return std::unexpected("TetraFeatures not initialized");
    }

    const size_t N = static_cast<size_t>(positions.shape()[0]);

    // If forward cache is not valid, compute activations on-the-fly
    // This is less efficient but allows backward to be called without prior forward_with_cache
    bool recompute_activations = !forward_cache_.valid;
    if (recompute_activations) {
        // Clear and initialize cache
        forward_cache_.layer_inputs.clear();
        forward_cache_.pre_activations.clear();

        int num_layers_local = mlp_config_.num_hidden_layers + 1;
        forward_cache_.layer_inputs.resize(static_cast<size_t>(num_layers_local));
        forward_cache_.pre_activations.resize(static_cast<size_t>(num_layers_local));

        // Compute hash encoding
        core::Tensor encoded = encode(positions);
        forward_cache_.hash_encoding = encoded.clone();
        forward_cache_.layer_inputs[0] = encoded.clone();

        // Allocate tensors for each layer
        for (int l = 0; l < num_layers_local; ++l) {
            int out_dim = (l == num_layers_local - 1) ? mlp_config_.output_dim : mlp_config_.hidden_dim;

            forward_cache_.pre_activations[static_cast<size_t>(l)] = core::Tensor::empty(
                {N, static_cast<size_t>(out_dim)},
                core::Device::CUDA, core::DataType::Float32);

            if (l < num_layers_local - 1) {
                forward_cache_.layer_inputs[static_cast<size_t>(l) + 1] = core::Tensor::empty(
                    {N, static_cast<size_t>(out_dim)},
                    core::Device::CUDA, core::DataType::Float32);
            }
        }

        // Run forward pass with caching
        std::vector<const float*> weight_ptrs;
        std::vector<const float*> bias_ptrs;
        std::vector<float*> layer_input_ptrs;
        std::vector<float*> pre_act_ptrs;

        for (const auto& w : mlp_weights_) {
            weight_ptrs.push_back(w.ptr<float>());
        }
        for (const auto& b : mlp_biases_) {
            bias_ptrs.push_back(b.ptr<float>());
        }
        for (auto& t : forward_cache_.layer_inputs) {
            layer_input_ptrs.push_back(t.ptr<float>());
        }
        for (auto& t : forward_cache_.pre_activations) {
            pre_act_ptrs.push_back(t.ptr<float>());
        }

        core::Tensor mlp_output = core::Tensor::empty(
            {N, static_cast<size_t>(mlp_config_.output_dim)},
            core::Device::CUDA, core::DataType::Float32);

        int enc_dim = hash_config_.num_levels * hash_config_.features_per_level;

        cuda::launch_mlp_forward_with_cache(
            encoded.ptr<float>(),
            weight_ptrs.data(),
            bias_ptrs.data(),
            mlp_output.ptr<float>(),
            layer_input_ptrs.data(),
            pre_act_ptrs.data(),
            N,
            enc_dim,
            mlp_config_.hidden_dim,
            mlp_config_.output_dim,
            mlp_config_.num_hidden_layers,
            mlp_config_.use_relu,
            nullptr);

        forward_cache_.mlp_output = std::move(mlp_output);
        forward_cache_.valid = true;
    }
    const size_t total_hash_entries = offsets_[hash_config_.num_levels];
    const int total_hash_features = hash_config_.num_levels * hash_config_.features_per_level;
    const int num_layers = mlp_config_.num_hidden_layers + 1;

    // Validate forward cache dimensions match
    if (forward_cache_.layer_inputs.size() != static_cast<size_t>(num_layers) ||
        forward_cache_.pre_activations.size() != static_cast<size_t>(num_layers)) {
        return std::unexpected("Forward cache has inconsistent layer count");
    }

    // Allocate gradient tensors
    core::Tensor grad_positions = core::Tensor::zeros(
        {N, 3}, core::Device::CUDA, core::DataType::Float32);

    core::Tensor grad_hash_table = core::Tensor::zeros(
        {total_hash_entries * hash_config_.features_per_level},
        core::Device::CUDA, core::DataType::Float32);

    // Get the gradient from the output (RGB)
    if (!grad_inputs.grad_rgb.is_valid()) {
        return std::unexpected("grad_rgb is required for backward pass");
    }

    core::Tensor grad_rgb = grad_inputs.grad_rgb.to(core::Device::CUDA);

    // ------------------------------------------------------------
    // MLP Backward Pass with proper ReLU gradient masking
    // ------------------------------------------------------------
    // For each layer l (in reverse):
    //   Forward: h_{l+1} = relu(z_l) where z_l = W_l @ h_l + b_l
    //   Backward:
    //     grad_b_l = sum(grad_z_l, axis=0)
    //     grad_W_l = h_l^T @ grad_z_l
    //     grad_h_l = grad_z_l @ W_l^T
    //     grad_z_{l-1} = grad_h_l * (z_{l-1} > 0) if l > 0 (ReLU mask)
    // ------------------------------------------------------------

    // Allocate gradient tensors for MLP weights and biases
    std::vector<core::Tensor> grad_mlp_weights;
    std::vector<core::Tensor> grad_mlp_biases;
    grad_mlp_weights.reserve(mlp_weights_.size());
    grad_mlp_biases.reserve(mlp_biases_.size());

    for (size_t l = 0; l < mlp_weights_.size(); ++l) {
        grad_mlp_weights.push_back(core::Tensor::zeros(
            {mlp_weights_[l].shape()[0], mlp_weights_[l].shape()[1]},
            core::Device::CUDA, core::DataType::Float32));

        grad_mlp_biases.push_back(core::Tensor::zeros(
            {mlp_biases_[l].shape()[0]},
            core::Device::CUDA, core::DataType::Float32));
    }

    // Start with grad_output = grad_rgb (only first 3 channels matter for RGB)
    // But we received grad_rgb for the sigmoid output, need to backprop through sigmoid
    // The sigmoid was applied externally, so grad_rgb is the gradient w.r.t. sigmoid output
    // grad_mlp_output = grad_rgb * sigmoid'(mlp_output) for RGB channels
    // For simplicity, we assume grad_rgb already accounts for sigmoid derivative
    // (This is typically handled by the loss function)

    // Initialize grad_z for the last layer (output layer has no activation)
    // grad_z_{L-1} = grad_output (assuming linear output for MLP)
    // Note: RGB extraction takes first 3 channels, so we need to pad
    core::Tensor grad_z = core::Tensor::zeros(
        {N, static_cast<size_t>(mlp_config_.output_dim)},
        core::Device::CUDA, core::DataType::Float32);

    // Copy grad_rgb to first 3 channels of grad_z
    cudaMemcpy2D(
        grad_z.ptr<float>(),
        static_cast<size_t>(mlp_config_.output_dim) * sizeof(float),
        grad_rgb.ptr<float>(),
        3 * sizeof(float),
        3 * sizeof(float),
        N,
        cudaMemcpyDeviceToDevice);

    // Backward through layers in reverse order
    for (int l = num_layers - 1; l >= 0; --l) {
        int in_features = static_cast<int>(mlp_weights_[static_cast<size_t>(l)].shape()[0]);
        int out_features = static_cast<int>(mlp_weights_[static_cast<size_t>(l)].shape()[1]);

        // Get layer input h_l from cache
        const core::Tensor& h_l = forward_cache_.layer_inputs[static_cast<size_t>(l)];

        // Compute grad_bias: grad_b_l = sum(grad_z, axis=0)
        cuda::launch_bias_grad(
            grad_z.ptr<float>(),
            grad_mlp_biases[static_cast<size_t>(l)].ptr<float>(),
            static_cast<int>(N),
            out_features,
            nullptr);

        // Compute grad_weight: grad_W_l = h_l^T @ grad_z
        // Weight layout is [in_features, out_features]
        cuda::launch_weight_grad(
            h_l.ptr<float>(),
            grad_z.ptr<float>(),
            grad_mlp_weights[static_cast<size_t>(l)].ptr<float>(),
            static_cast<int>(N),
            in_features,
            out_features,
            nullptr);

        // Compute grad_h_l = grad_z @ W_l^T if not the first layer
        if (l > 0) {
            core::Tensor grad_h = core::Tensor::zeros(
                {N, static_cast<size_t>(in_features)},
                core::Device::CUDA, core::DataType::Float32);

            cuda::launch_matmul_backward(
                grad_z.ptr<float>(),
                mlp_weights_[static_cast<size_t>(l)].ptr<float>(),
                grad_h.ptr<float>(),
                static_cast<int>(N),
                out_features,
                in_features,
                nullptr);

            // Apply ReLU derivative mask: grad_z_{l-1} = grad_h * (z_{l-1} > 0)
            // z_{l-1} is the pre-activation of the previous layer
            const core::Tensor& z_prev = forward_cache_.pre_activations[static_cast<size_t>(l) - 1];

            core::Tensor grad_z_prev = core::Tensor::zeros(
                {N, static_cast<size_t>(in_features)},
                core::Device::CUDA, core::DataType::Float32);

            if (mlp_config_.use_relu) {
                cuda::launch_relu_backward(
                    grad_h.ptr<float>(),
                    z_prev.ptr<float>(),
                    grad_z_prev.ptr<float>(),
                    N * static_cast<size_t>(in_features),
                    nullptr);
            } else {
                // No activation, gradient passes through
                grad_z_prev = std::move(grad_h);
            }

            grad_z = std::move(grad_z_prev);
        } else {
            // First layer: compute grad_hash_encoding
            core::Tensor grad_h = core::Tensor::zeros(
                {N, static_cast<size_t>(in_features)},
                core::Device::CUDA, core::DataType::Float32);

            cuda::launch_matmul_backward(
                grad_z.ptr<float>(),
                mlp_weights_[0].ptr<float>(),
                grad_h.ptr<float>(),
                static_cast<int>(N),
                out_features,
                in_features,
                nullptr);

            grad_z = std::move(grad_h);
        }
    }

    // grad_z now contains grad_hash_encoding (gradient w.r.t. MLP input)
    core::Tensor grad_hash_encoding = std::move(grad_z);

    // ------------------------------------------------------------
    // Hash Grid Backward Pass
    // ------------------------------------------------------------
    const size_t table_size = static_cast<size_t>(1) << hash_config_.log2_hashmap_size;

    // Upload auxiliary data to GPU
    core::Tensor resolutions_tensor = core::Tensor::empty(
        {static_cast<size_t>(hash_config_.num_levels)},
        core::Device::CPU, core::DataType::Float32);
    std::memcpy(resolutions_tensor.ptr<float>(), resolutions_.data(),
                static_cast<size_t>(hash_config_.num_levels) * sizeof(float));
    resolutions_tensor = resolutions_tensor.to(core::Device::CUDA);

    core::Tensor offsets_tensor = core::Tensor::empty(
        {static_cast<size_t>(hash_config_.num_levels + 1)},
        core::Device::CPU, core::DataType::Int64);
    for (int i = 0; i <= hash_config_.num_levels; ++i) {
        offsets_tensor.ptr<int64_t>()[i] = static_cast<int64_t>(offsets_[static_cast<size_t>(i)]);
    }
    offsets_tensor = offsets_tensor.to(core::Device::CUDA);

    // Hash grid backward: compute grad_hash_table
    cuda::launch_hash_grid_backward(
        positions.ptr<float>(),
        grad_hash_encoding.ptr<float>(),
        scene_aabb_.ptr<float>(),
        grad_hash_table.ptr<float>(),
        resolutions_tensor.ptr<float>(),
        reinterpret_cast<const size_t*>(offsets_tensor.ptr<int64_t>()),
        N,
        hash_config_.num_levels,
        hash_config_.features_per_level,
        table_size,
        nullptr);

    // Hash grid position backward: compute grad_positions
    cuda::launch_hash_grid_position_backward(
        positions.ptr<float>(),
        grad_hash_encoding.ptr<float>(),
        hash_table_.ptr<float>(),
        scene_aabb_.ptr<float>(),
        grad_positions.ptr<float>(),
        resolutions_tensor.ptr<float>(),
        reinterpret_cast<const size_t*>(offsets_tensor.ptr<int64_t>()),
        N,
        hash_config_.num_levels,
        hash_config_.features_per_level,
        table_size,
        nullptr);

    // ------------------------------------------------------------
    // Flatten all gradients into grad_mlp_params
    // ------------------------------------------------------------
    core::Tensor grad_all_params = core::Tensor::zeros(
        {num_parameters()}, core::Device::CUDA, core::DataType::Float32);

    // Copy hash table gradient to beginning
    size_t hash_params = total_hash_entries * static_cast<size_t>(hash_config_.features_per_level);
    size_t offset = 0;

    cudaMemcpy(
        grad_all_params.ptr<float>() + offset,
        grad_hash_table.ptr<float>(),
        hash_params * sizeof(float),
        cudaMemcpyDeviceToDevice);
    offset += hash_params;

    // Copy MLP weight gradients
    for (const auto& gw : grad_mlp_weights) {
        size_t numel = static_cast<size_t>(gw.numel());
        cudaMemcpy(
            grad_all_params.ptr<float>() + offset,
            gw.ptr<float>(),
            numel * sizeof(float),
            cudaMemcpyDeviceToDevice);
        offset += numel;
    }

    // Copy MLP bias gradients
    for (const auto& gb : grad_mlp_biases) {
        size_t numel = static_cast<size_t>(gb.numel());
        cudaMemcpy(
            grad_all_params.ptr<float>() + offset,
            gb.ptr<float>(),
            numel * sizeof(float),
            cudaMemcpyDeviceToDevice);
        offset += numel;
    }

    return FeatureBackwardResult{
        std::move(grad_positions),
        std::move(grad_hash_table),
        std::move(grad_all_params),
        std::move(grad_mlp_weights),
        std::move(grad_mlp_biases)
    };
}

// ------------------------------
// PARAMETER ACCESS
// ------------------------------

std::vector<core::Tensor*> TetraFeatures::parameters() {
    std::vector<core::Tensor*> params;
    params.push_back(&hash_table_);
    for (auto& w : mlp_weights_) {
        params.push_back(&w);
    }
    for (auto& b : mlp_biases_) {
        params.push_back(&b);
    }
    return params;
}

std::vector<core::Tensor*> TetraFeatures::mlp_weights() {
    std::vector<core::Tensor*> params;
    for (auto& w : mlp_weights_) {
        params.push_back(&w);
    }
    for (auto& b : mlp_biases_) {
        params.push_back(&b);
    }
    return params;
}

size_t TetraFeatures::num_parameters() const {
    size_t count = 0;
    if (hash_table_.is_valid()) {
        count += static_cast<size_t>(hash_table_.numel());
    }
    for (const auto& w : mlp_weights_) {
        count += static_cast<size_t>(w.numel());
    }
    for (const auto& b : mlp_biases_) {
        count += static_cast<size_t>(b.numel());
    }
    return count;
}

// ------------------------------
// SERIALIZATION
// ------------------------------

void TetraFeatures::serialize(std::ostream& os) const {
    // Write magic and version
    const uint32_t magic = 0x46454154;  // "FEAT"
    const uint32_t version = 1;
    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write configs
    os.write(reinterpret_cast<const char*>(&hash_config_), sizeof(hash_config_));
    os.write(reinterpret_cast<const char*>(&mlp_config_), sizeof(mlp_config_));

    // Write tensors using stream operator
    os << scene_aabb_;
    os << hash_table_;

    size_t num_layers = mlp_weights_.size();
    os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    for (const auto& w : mlp_weights_) {
        os << w;
    }
    for (const auto& b : mlp_biases_) {
        os << b;
    }
}

void TetraFeatures::deserialize(std::istream& is) {
    // Read and verify magic
    uint32_t magic;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x46454154) {
        throw std::runtime_error("Invalid TetraFeatures file format");
    }

    uint32_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported TetraFeatures version");
    }

    // Read configs
    is.read(reinterpret_cast<char*>(&hash_config_), sizeof(hash_config_));
    is.read(reinterpret_cast<char*>(&mlp_config_), sizeof(mlp_config_));

    // Read tensors using stream operator
    is >> scene_aabb_;
    is >> hash_table_;

    size_t num_layers;
    is.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    mlp_weights_.resize(num_layers);
    mlp_biases_.resize(num_layers);

    for (auto& w : mlp_weights_) {
        is >> w;
    }
    for (auto& b : mlp_biases_) {
        is >> b;
    }

    // Recompute resolutions and offsets
    resolutions_.resize(hash_config_.num_levels);
    offsets_.resize(hash_config_.num_levels + 1);

    size_t total_entries = 0;
    float b = hash_config_.per_level_scale;

    for (int l = 0; l < hash_config_.num_levels; ++l) {
        float res = std::floor(
            static_cast<float>(hash_config_.base_resolution) * std::pow(b, static_cast<float>(l)));
        res = std::min(res, static_cast<float>(hash_config_.max_resolution));
        resolutions_[l] = res;

        size_t table_size = static_cast<size_t>(1) << hash_config_.log2_hashmap_size;
        offsets_[l] = total_entries;
        total_entries += table_size;
    }
    offsets_[hash_config_.num_levels] = total_entries;

    initialized_ = true;
}

} // namespace lfs::tetra
