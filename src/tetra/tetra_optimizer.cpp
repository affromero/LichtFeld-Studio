/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_optimizer.hpp"
#include "adam_api.h" // fast_lfs::optimizer::adam_step_raw
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(err));     \
        }                                                          \
    } while (0)

namespace lfs::tetra {

namespace {
    constexpr uint32_t TETRA_OPT_MAGIC = 0x4C465445; // "LFTE"
    constexpr uint32_t TETRA_OPT_VERSION = 2;        // v2: added pertet_lr_
} // namespace

// ------------------------------
// LIFECYCLE
// ------------------------------

TetraOptimizer::TetraOptimizer(TetraMesh& mesh,
                               TetraFeatures& features,
                               const TetraOptimizerConfig& config)
    : config_(config)
    , mesh_(mesh)
    , features_(features) {

    // Scale vertex LR by scene scale (as in Python reference)
    vert_lr_multi_ = mesh_.get_scene_scale();
    vertices_lr_ = vert_lr_multi_ * config_.vertices_lr;
    encoding_lr_ = config_.encoding_lr;
    network_lr_ = config_.network_lr;
    pertet_lr_ = config_.pertet_lr;

    LOG_DEBUG("TetraOptimizer created: vert_lr_multi={:.4f}", vert_lr_multi_);
}

// ------------------------------
// CORE OPTIMIZER INTERFACE
// ------------------------------

void TetraOptimizer::step(const int iteration) {
    // Update learning rates first
    update_learning_rate(iteration);

    // Check if mesh should be frozen
    if (iteration >= config_.freeze_start && !frozen_) {
        frozen_ = true;
        LOG_INFO("Freezing mesh connectivity at iteration {}", iteration);
    }

    // Step each parameter group
    step_param(TetraParamType::Vertices, iteration);
    step_param(TetraParamType::Encoding, iteration);
    step_param(TetraParamType::Network, iteration);

    // Step per-tetrahedron parameters
    step_param(TetraParamType::Density, iteration);
    step_param(TetraParamType::BaseColor, iteration);
    step_param(TetraParamType::Gradient, iteration);

    current_iteration_ = iteration;
}

void TetraOptimizer::zero_grad() {
    for (auto& [_, state] : states_) {
        if (state.grad.is_valid() && state.grad.numel() > 0) {
            const size_t bytes = state.size *
                                 (state.grad.numel() / state.grad.shape()[0]) *
                                 sizeof(float);
            CHECK_CUDA(cudaMemsetAsync(state.grad.ptr<float>(), 0, bytes, nullptr));
        }
    }
}

// ------------------------------
// LEARNING RATE MANAGEMENT
// ------------------------------

void TetraOptimizer::update_learning_rate(const int iteration) {
    // Vertex learning rate: exponential decay with spike
    const float base_vert_lr = compute_exponential_lr(
        vert_lr_multi_ * config_.vertices_lr,
        vert_lr_multi_ * config_.vertices_lr_final,
        config_.lr_delay_mult,
        config_.lr_delay_steps,
        config_.freeze_start,
        iteration);

    if (config_.enable_lr_spikes) {
        vertices_lr_ = apply_lr_spike(
            base_vert_lr,
            vert_lr_multi_ * config_.vertices_lr,
            iteration);
    } else {
        vertices_lr_ = base_vert_lr;
    }

    // Encoding learning rate: exponential decay with spike
    const float base_enc_lr = compute_exponential_lr(
        config_.encoding_lr,
        config_.encoding_lr_final,
        config_.lr_delay_mult,
        config_.lr_delay_steps,
        config_.freeze_start,
        iteration);

    if (config_.enable_lr_spikes) {
        encoding_lr_ = apply_lr_spike(
            base_enc_lr,
            config_.encoding_lr,
            iteration);
    } else {
        encoding_lr_ = base_enc_lr;
    }

    // Network learning rate: exponential decay with spike
    const float base_net_lr = compute_exponential_lr(
        config_.network_lr,
        config_.network_lr_final,
        config_.lr_delay_mult,
        config_.lr_delay_steps,
        config_.freeze_start,
        iteration);

    if (config_.enable_lr_spikes) {
        network_lr_ = apply_lr_spike(
            base_net_lr,
            config_.network_lr,
            iteration);
    } else {
        network_lr_ = base_net_lr;
    }

    // Per-tetrahedron learning rate: exponential decay (no spike for per-tet params)
    pertet_lr_ = compute_exponential_lr(
        config_.pertet_lr,
        config_.pertet_lr_final,
        config_.lr_delay_mult,
        config_.lr_delay_steps,
        config_.freeze_start,
        iteration);

    // Update state current_lr values
    if (auto* state = get_state(TetraParamType::Vertices)) {
        const_cast<TetraParamState*>(state)->current_lr = vertices_lr_;
    }
    if (auto* state = get_state(TetraParamType::Encoding)) {
        const_cast<TetraParamState*>(state)->current_lr = encoding_lr_;
    }
    if (auto* state = get_state(TetraParamType::Network)) {
        const_cast<TetraParamState*>(state)->current_lr = network_lr_;
    }
    // Per-tetrahedron parameters share the same learning rate
    if (auto* state = get_state(TetraParamType::Density)) {
        const_cast<TetraParamState*>(state)->current_lr = pertet_lr_;
    }
    if (auto* state = get_state(TetraParamType::BaseColor)) {
        const_cast<TetraParamState*>(state)->current_lr = pertet_lr_;
    }
    if (auto* state = get_state(TetraParamType::Gradient)) {
        const_cast<TetraParamState*>(state)->current_lr = pertet_lr_;
    }
}

float TetraOptimizer::compute_exponential_lr(
    const float lr_init,
    const float lr_final,
    const float lr_delay_mult,
    const int lr_delay_steps,
    const int max_steps,
    const int iteration) const {

    if (iteration >= max_steps) {
        return lr_final;
    }

    // Compute delay factor for warmup
    float delay_factor = 1.0f;
    if (lr_delay_steps > 0 && iteration < lr_delay_steps) {
        const float t = static_cast<float>(iteration) / static_cast<float>(lr_delay_steps);
        delay_factor = lr_delay_mult + (1.0f - lr_delay_mult) * t;
    }

    // Exponential interpolation: lr_init * (lr_final/lr_init)^(t/max_t)
    // Handle lr_init=0 case to avoid NaN
    if (lr_init <= 0.0f) {
        return 0.0f;
    }
    const float t = static_cast<float>(iteration) / static_cast<float>(max_steps);
    const float ratio = lr_final / lr_init;
    const float lr = lr_init * std::pow(ratio, t);

    return lr * delay_factor;
}

float TetraOptimizer::apply_lr_spike(
    const float base_lr,
    const float spike_lr,
    const int iteration) const {

    // Only apply spikes during densification window
    if (iteration < config_.spike_midpoint || iteration >= config_.densify_end) {
        return base_lr;
    }

    // Check if we're in a spike window (after densification)
    const int offset_from_densify = iteration % config_.densify_interval;
    if (offset_from_densify < config_.spike_duration) {
        // Linear decay from spike_lr to base_lr over spike duration
        const float spike_progress = static_cast<float>(offset_from_densify) /
                                     static_cast<float>(config_.spike_duration);
        return spike_lr * (1.0f - spike_progress) + base_lr * spike_progress;
    }

    return base_lr;
}

float TetraOptimizer::get_lr(TetraParamType type) const {
    switch (type) {
    case TetraParamType::Vertices: return vertices_lr_;
    case TetraParamType::Encoding: return encoding_lr_;
    case TetraParamType::Network: return network_lr_;
    case TetraParamType::Density:
    case TetraParamType::BaseColor:
    case TetraParamType::Gradient: return pertet_lr_;
    }
    return config_.vertices_lr;
}

void TetraOptimizer::set_lr(TetraParamType type, float lr) {
    switch (type) {
    case TetraParamType::Vertices: vertices_lr_ = lr; break;
    case TetraParamType::Encoding: encoding_lr_ = lr; break;
    case TetraParamType::Network: network_lr_ = lr; break;
    case TetraParamType::Density:
    case TetraParamType::BaseColor:
    case TetraParamType::Gradient: pertet_lr_ = lr; break;
    }
}

// ------------------------------
// GRADIENT ACCESS
// ------------------------------

void TetraOptimizer::allocate_gradients(const size_t capacity) {
    const std::array<TetraParamType, 6> types = {
        TetraParamType::Vertices,
        TetraParamType::Encoding,
        TetraParamType::Network,
        TetraParamType::Density,
        TetraParamType::BaseColor,
        TetraParamType::Gradient
    };

    for (const auto type : types) {
        auto& param = get_param(type);
        const auto name = param_name(type);

        if (!param.is_valid()) {
            states_[name] = TetraParamState{};
            continue;
        }

        auto& state = states_[name];
        const size_t param_size = static_cast<size_t>(param.shape()[0]);
        const size_t alloc_cap = (capacity > param_size) ? capacity : param_size;

        if (alloc_cap > param_size) {
            state.grad = core::Tensor::zeros_direct(param.shape(), alloc_cap);
            state.exp_avg = core::Tensor::zeros_direct(param.shape(), alloc_cap);
            state.exp_avg_sq = core::Tensor::zeros_direct(param.shape(), alloc_cap);
        } else {
            state.grad = core::Tensor::zeros(param.shape(), param.device());
            state.exp_avg = core::Tensor::zeros(param.shape(), param.device());
            state.exp_avg_sq = core::Tensor::zeros(param.shape(), param.device());
        }

        state.capacity = alloc_cap;
        state.size = param_size;
        state.step_count = 0;
        state.current_lr = get_lr(type);

        LOG_DEBUG("Allocated gradients for {}: size={}, capacity={}",
                  name, param_size, alloc_cap);
    }
}

bool TetraOptimizer::has_gradients() const {
    for (const auto& [_, state] : states_) {
        if (state.grad.is_valid() && state.grad.numel() > 0) {
            return true;
        }
    }
    return false;
}

core::Tensor& TetraOptimizer::get_grad(TetraParamType type) {
    const auto name = param_name(type);
    const auto it = states_.find(name);
    if (it == states_.end()) {
        throw std::runtime_error("get_grad: " + name + " not initialized");
    }
    return it->second.grad;
}

const core::Tensor& TetraOptimizer::get_grad(TetraParamType type) const {
    const auto name = param_name(type);
    const auto it = states_.find(name);
    if (it == states_.end()) {
        throw std::runtime_error("get_grad: " + name + " not initialized");
    }
    return it->second.grad;
}

// ------------------------------
// VERTEX MANAGEMENT
// ------------------------------

std::expected<void, std::string> TetraOptimizer::add_vertices(
    const core::Tensor& new_verts) {

    if (frozen_) {
        return std::unexpected("Cannot add vertices: mesh is frozen");
    }

    if (!new_verts.is_valid() || new_verts.ndim() != 2 || new_verts.shape()[1] != 3) {
        return std::unexpected("Invalid new_verts shape: expected [N, 3]");
    }

    // Concatenate new vertices to mesh
    auto& vertices = mesh_.vertices();
    if (!vertices.is_valid()) {
        return std::unexpected("Mesh vertices not initialized");
    }

    const size_t n_new = static_cast<size_t>(new_verts.shape()[0]);

    // Concatenate vertices
    vertices = core::Tensor::cat({vertices, new_verts}, 0);

    // Extend optimizer state
    extend_state_for_new_params(TetraParamType::Vertices, n_new);

    LOG_DEBUG("Added {} vertices, new total: {}", n_new, vertices.shape()[0]);

    return {};
}

std::expected<void, std::string> TetraOptimizer::remove_vertices(
    const core::Tensor& keep_mask) {

    if (frozen_) {
        return std::unexpected("Cannot remove vertices: mesh is frozen");
    }

    if (!keep_mask.is_valid() || keep_mask.ndim() != 1) {
        return std::unexpected("Invalid keep_mask shape: expected [V]");
    }

    auto& vertices = mesh_.vertices();
    const size_t n_verts = static_cast<size_t>(vertices.shape()[0]);

    if (static_cast<size_t>(keep_mask.shape()[0]) != n_verts) {
        return std::unexpected("keep_mask size mismatch with vertex count");
    }

    // Select kept vertices
    const core::Tensor indices = keep_mask.nonzero().squeeze(1);
    vertices = vertices.index_select(0, indices);

    // Compact optimizer state
    compact_state(TetraParamType::Vertices, keep_mask);

    LOG_DEBUG("Removed vertices, new total: {}", vertices.shape()[0]);

    return {};
}

void TetraOptimizer::reset_state_at_indices(const std::vector<int64_t>& indices) {
    if (indices.empty()) {
        return;
    }

    const auto name = param_name(TetraParamType::Vertices);
    const auto it = states_.find(name);
    if (it == states_.end()) {
        return;
    }

    auto& state = it->second;

    if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid()) {
        LOG_WARN("reset_state_at_indices: {} state tensors invalid", name);
        return;
    }

    const auto& shape = state.exp_avg.shape();
    int row_size = 1;
    for (size_t i = 1; i < shape.rank(); i++) {
        row_size *= static_cast<int>(shape[i]);
    }

    // Upload indices to GPU
    int64_t* d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, indices.size() * sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(d_indices, indices.data(),
                          indices.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    // Zero optimizer state at indices
    fast_lfs::optimizer::zero_rows_at_indices(
        state.exp_avg.ptr<float>(), d_indices, indices.size(), row_size);
    fast_lfs::optimizer::zero_rows_at_indices(
        state.exp_avg_sq.ptr<float>(), d_indices, indices.size(), row_size);

    CHECK_CUDA(cudaFree(d_indices));
}

// ------------------------------
// STATE ACCESS
// ------------------------------

const TetraParamState* TetraOptimizer::get_state(TetraParamType type) const {
    const auto name = param_name(type);
    const auto it = states_.find(name);
    return (it != states_.end()) ? &it->second : nullptr;
}

int64_t TetraOptimizer::get_step_count(TetraParamType type) const {
    const auto* state = get_state(type);
    return state ? state->step_count : 0;
}

// ------------------------------
// SERIALIZATION
// ------------------------------

void TetraOptimizer::serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&TETRA_OPT_MAGIC), sizeof(TETRA_OPT_MAGIC));
    os.write(reinterpret_cast<const char*>(&TETRA_OPT_VERSION), sizeof(TETRA_OPT_VERSION));

    // Write config
    os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));

    // Write current state
    os.write(reinterpret_cast<const char*>(&vertices_lr_), sizeof(vertices_lr_));
    os.write(reinterpret_cast<const char*>(&encoding_lr_), sizeof(encoding_lr_));
    os.write(reinterpret_cast<const char*>(&network_lr_), sizeof(network_lr_));
    os.write(reinterpret_cast<const char*>(&pertet_lr_), sizeof(pertet_lr_));
    os.write(reinterpret_cast<const char*>(&vert_lr_multi_), sizeof(vert_lr_multi_));
    os.write(reinterpret_cast<const char*>(&current_iteration_), sizeof(current_iteration_));
    os.write(reinterpret_cast<const char*>(&frozen_), sizeof(frozen_));

    // Write optimizer states
    uint32_t num_states = 0;
    for (const auto& [_, state] : states_) {
        if (state.exp_avg.is_valid() && state.exp_avg_sq.is_valid()) {
            ++num_states;
        }
    }
    os.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));

    for (const auto& [name, state] : states_) {
        if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid()) {
            continue;
        }

        const auto name_len = static_cast<uint32_t>(name.size());
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        os.write(name.data(), name_len);
        os.write(reinterpret_cast<const char*>(&state.step_count), sizeof(state.step_count));
        os.write(reinterpret_cast<const char*>(&state.capacity), sizeof(state.capacity));
        os.write(reinterpret_cast<const char*>(&state.size), sizeof(state.size));
        os.write(reinterpret_cast<const char*>(&state.current_lr), sizeof(state.current_lr));
        os << state.exp_avg << state.exp_avg_sq;
    }

    LOG_DEBUG("Serialized TetraOptimizer: {} states", num_states);
}

void TetraOptimizer::deserialize(std::istream& is) {
    uint32_t magic, version;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != TETRA_OPT_MAGIC) {
        throw std::runtime_error("Invalid TetraOptimizer checkpoint magic");
    }
    if (version != TETRA_OPT_VERSION) {
        throw std::runtime_error("Unsupported TetraOptimizer checkpoint version");
    }

    // Read config
    is.read(reinterpret_cast<char*>(&config_), sizeof(config_));

    // Read current state
    is.read(reinterpret_cast<char*>(&vertices_lr_), sizeof(vertices_lr_));
    is.read(reinterpret_cast<char*>(&encoding_lr_), sizeof(encoding_lr_));
    is.read(reinterpret_cast<char*>(&network_lr_), sizeof(network_lr_));
    is.read(reinterpret_cast<char*>(&pertet_lr_), sizeof(pertet_lr_));
    is.read(reinterpret_cast<char*>(&vert_lr_multi_), sizeof(vert_lr_multi_));
    is.read(reinterpret_cast<char*>(&current_iteration_), sizeof(current_iteration_));
    is.read(reinterpret_cast<char*>(&frozen_), sizeof(frozen_));

    // Read optimizer states
    uint32_t num_states;
    is.read(reinterpret_cast<char*>(&num_states), sizeof(num_states));

    states_.clear();
    for (uint32_t i = 0; i < num_states; ++i) {
        uint32_t name_len;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        is.read(name.data(), name_len);

        TetraParamState state;
        is.read(reinterpret_cast<char*>(&state.step_count), sizeof(state.step_count));
        is.read(reinterpret_cast<char*>(&state.capacity), sizeof(state.capacity));
        is.read(reinterpret_cast<char*>(&state.size), sizeof(state.size));
        is.read(reinterpret_cast<char*>(&state.current_lr), sizeof(state.current_lr));

        is >> state.exp_avg >> state.exp_avg_sq;
        state.exp_avg = state.exp_avg.cuda();
        state.exp_avg_sq = state.exp_avg_sq.cuda();

        // Allocate gradient buffer
        if (state.exp_avg.is_valid()) {
            state.grad = core::Tensor::zeros_direct(state.exp_avg.shape(), state.capacity);
        }

        states_[name] = std::move(state);
    }

    LOG_DEBUG("Deserialized TetraOptimizer: {} states", num_states);
}

// ------------------------------
// INTERNAL HELPERS
// ------------------------------

core::Tensor& TetraOptimizer::get_param(TetraParamType type) {
    switch (type) {
    case TetraParamType::Vertices:
        return mesh_.vertices();
    case TetraParamType::Encoding:
        return features_.hash_table();
    case TetraParamType::Network: {
        // Return first MLP weight as representative
        auto weights = features_.mlp_weights();
        if (!weights.empty() && weights[0] != nullptr) {
            return *weights[0];
        }
        throw std::runtime_error("Network parameters not initialized");
    }
    case TetraParamType::Density:
        return mesh_.density();
    case TetraParamType::BaseColor:
        return mesh_.base_color();
    case TetraParamType::Gradient:
        return mesh_.gradient();
    }
    throw std::runtime_error("Invalid TetraParamType");
}

std::string TetraOptimizer::param_name(TetraParamType type) {
    switch (type) {
    case TetraParamType::Vertices: return "vertices";
    case TetraParamType::Encoding: return "encoding";
    case TetraParamType::Network: return "network";
    case TetraParamType::Density: return "density";
    case TetraParamType::BaseColor: return "base_color";
    case TetraParamType::Gradient: return "gradient";
    }
    return "unknown";
}

void TetraOptimizer::init_state(TetraParamType type) {
    auto& param = get_param(type);
    const auto name = param_name(type);

    if (!param.is_valid()) {
        throw std::runtime_error("init_state: " + name + " not valid");
    }
    if (param.ndim() == 0) {
        throw std::runtime_error("init_state: " + name + " has rank 0");
    }

    auto& state = states_[name];
    const size_t param_size = static_cast<size_t>(param.shape()[0]);

    // Allocate with some growth room
    const size_t initial_cap = static_cast<size_t>(param_size * 1.5f);

    if (!state.grad.is_valid() || state.grad.numel() == 0) {
        state.grad = (initial_cap > param_size)
                         ? core::Tensor::zeros_direct(param.shape(), initial_cap)
                         : core::Tensor::zeros(param.shape(), param.device());
    }

    if (initial_cap > param_size) {
        state.exp_avg = core::Tensor::zeros_direct(param.shape(), initial_cap);
        state.exp_avg_sq = core::Tensor::zeros_direct(param.shape(), initial_cap);
        state.capacity = initial_cap;
    } else {
        state.exp_avg = core::Tensor::zeros(param.shape(), param.device());
        state.exp_avg_sq = core::Tensor::zeros(param.shape(), param.device());
        state.capacity = param_size;
    }

    state.size = param_size;
    state.step_count = 0;
    state.current_lr = get_lr(type);

    LOG_DEBUG("Initialized optimizer state for {}: size={}, capacity={}",
              name, param_size, state.capacity);
}

void TetraOptimizer::step_param(TetraParamType type, const int iteration) {
    auto& param = get_param(type);
    if (!param.is_valid() || param.numel() == 0) {
        return;
    }

    // Skip vertex optimization if frozen
    if (type == TetraParamType::Vertices && frozen_) {
        return;
    }

    const auto name = param_name(type);
    if (!states_.contains(name)) {
        init_state(type);
    }

    auto& state = states_[name];
    if (!state.grad.is_valid() || state.grad.numel() == 0 ||
        !state.exp_avg.is_valid() || state.exp_avg.numel() == 0) {
        return;
    }

    state.step_count++;

    // Compute bias correction factors
    const double bias_correction1_rcp =
        1.0 / (1.0 - std::pow(static_cast<double>(config_.beta1), state.step_count));
    const double bias_correction2_sqrt_rcp =
        1.0 / std::sqrt(1.0 - std::pow(static_cast<double>(config_.beta2), state.step_count));

    const float param_lr = get_lr(type);

    const size_t param_size = static_cast<size_t>(param.shape()[0]);
    if (param_size != state.size) {
        // Per-tet parameters can be resized when mesh is re-triangulated
        // Reinitialize state for these parameter types
        const bool is_pertet = (type == TetraParamType::Density ||
                                type == TetraParamType::BaseColor ||
                                type == TetraParamType::Gradient);
        if (is_pertet) {
            LOG_DEBUG("Reinitializing {} state: size changed from {} to {}",
                      name, state.size, param_size);
            init_state(type);
        } else {
            throw std::runtime_error("Optimizer state desync: " + name);
        }
    }

    const size_t feature_dim = param.numel() / param_size;
    const size_t num_elements = state.size * feature_dim;

    LOG_INFO("step_param({}): param_size={}, feature_dim={}, num_elements={}, lr={}",
              name, param_size, feature_dim, num_elements, param_lr);

    if (num_elements == 0) {
        LOG_WARN("step_param({}): skipping - num_elements is 0", name);
        return;
    }

    // Validate pointers before kernel call
    LOG_INFO("step_param({}): validating pointers - param={}, exp_avg={}, exp_avg_sq={}, grad={}",
             name,
             param.is_valid() ? "valid" : "INVALID",
             state.exp_avg.is_valid() ? "valid" : "INVALID",
             state.exp_avg_sq.is_valid() ? "valid" : "INVALID",
             state.grad.is_valid() ? "valid" : "INVALID");

    // Call optimized CUDA Adam kernel
    LOG_INFO("step_param({}): calling adam_step_raw...", name);
    fast_lfs::optimizer::adam_step_raw(
        param.ptr<float>(),
        state.exp_avg.ptr<float>(),
        state.exp_avg_sq.ptr<float>(),
        state.grad.ptr<float>(),
        static_cast<int>(num_elements),
        param_lr,
        config_.beta1,
        config_.beta2,
        config_.eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);
}

void TetraOptimizer::extend_state_for_new_params(TetraParamType type, const size_t n_new) {
    const auto name = param_name(type);
    if (!states_.contains(name)) {
        LOG_DEBUG("extend_state_for_new_params({}): state not found, skipping", name);
        return;
    }

    auto& param = get_param(type);
    auto& state = states_[name];
    const size_t new_size = state.size + n_new;

    if (!param.is_valid() || param.shape().rank() == 0) {
        throw std::runtime_error("extend_state: " + name + " invalid");
    }
    if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
        throw std::runtime_error("extend_state: " + name + " state invalid");
    }

    // Fast path: use reserved capacity
    const bool all_have_capacity = state.grad.capacity() > 0 &&
                                   state.exp_avg.capacity() > 0 &&
                                   state.exp_avg_sq.capacity() > 0;
    const bool fits_in_capacity = new_size <= state.grad.capacity() &&
                                  new_size <= state.exp_avg.capacity() &&
                                  new_size <= state.exp_avg_sq.capacity();

    if (all_have_capacity && fits_in_capacity) {
        state.grad.append_zeros(n_new);
        state.exp_avg.append_zeros(n_new);
        state.exp_avg_sq.append_zeros(n_new);
        state.size = new_size;
        state.capacity = state.exp_avg.capacity();
        LOG_DEBUG("extend_state_for_new_params({}): fast path, new size = {}", name, new_size);
        return;
    }

    LOG_WARN("extend_state_for_new_params({}): SLOW PATH triggered", name);

    // Slow path: reallocate
    const auto& shape = param.shape();
    std::vector<size_t> new_dims(shape.rank());
    new_dims[0] = new_size;
    for (size_t i = 1; i < shape.rank(); i++) {
        new_dims[i] = shape[i];
    }

    const auto tensor_shape = core::TensorShape(new_dims);
    state.grad = core::Tensor::zeros(tensor_shape, param.device());
    auto new_exp_avg = core::Tensor::empty(tensor_shape, param.device());
    auto new_exp_avg_sq = core::Tensor::empty(tensor_shape, param.device());

    // Copy old data
    if (state.size > 0 && state.exp_avg.numel() > 0) {
        const size_t old_bytes = state.exp_avg.numel() * sizeof(float);
        CHECK_CUDA(cudaMemcpyAsync(new_exp_avg.ptr<float>(),
                                   state.exp_avg.ptr<float>(),
                                   old_bytes, cudaMemcpyDeviceToDevice, nullptr));
        CHECK_CUDA(cudaMemcpyAsync(new_exp_avg_sq.ptr<float>(),
                                   state.exp_avg_sq.ptr<float>(),
                                   old_bytes, cudaMemcpyDeviceToDevice, nullptr));
    }

    // Zero new rows
    const size_t row_size = param.numel() / shape[0];
    const size_t offset = state.exp_avg.numel() * sizeof(float);
    const size_t new_bytes = n_new * row_size * sizeof(float);
    CHECK_CUDA(cudaMemsetAsync(
        reinterpret_cast<char*>(new_exp_avg.ptr<float>()) + offset, 0, new_bytes, nullptr));
    CHECK_CUDA(cudaMemsetAsync(
        reinterpret_cast<char*>(new_exp_avg_sq.ptr<float>()) + offset, 0, new_bytes, nullptr));

    state.exp_avg = std::move(new_exp_avg);
    state.exp_avg_sq = std::move(new_exp_avg_sq);
    state.size = new_size;
    state.capacity = 0;
}

void TetraOptimizer::compact_state(TetraParamType type, const core::Tensor& keep_mask) {
    const auto name = param_name(type);
    if (!states_.contains(name)) {
        return;
    }

    auto& state = states_[name];
    if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid()) {
        return;
    }

    // Get indices to keep
    const core::Tensor indices = keep_mask.nonzero().squeeze(1);

    // Select kept rows
    state.exp_avg = state.exp_avg.index_select(0, indices);
    state.exp_avg_sq = state.exp_avg_sq.index_select(0, indices);

    // Reallocate gradient
    state.grad = core::Tensor::zeros(state.exp_avg.shape(), state.exp_avg.device());

    state.size = static_cast<size_t>(state.exp_avg.shape()[0]);
    state.capacity = state.size;

    LOG_DEBUG("compact_state({}): new size = {}", name, state.size);
}

} // namespace lfs::tetra
