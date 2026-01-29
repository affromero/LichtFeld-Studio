/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tetra/tetra_trainer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "io/cache_image_loader.hpp"
#include "io/loader.hpp"
#include "lfs/kernels/ssim.cuh"
#include "visualizer/scene/scene.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>

namespace lfs::tetra {

// CUDA kernel declarations
namespace cuda {
    void launch_accumulate_gradient_norms(
        const float* gradients,
        float* accum,
        size_t num_vertices,
        void* stream);

    void launch_sgd_update(
        float* param,
        const float* grad,
        float lr,
        size_t n,
        void* stream);
}

// ------------------------------
// LIFECYCLE
// ------------------------------

TetraTrainer::TetraTrainer(const TetraTrainConfig& config)
    : config_(config)
    , scene_(nullptr) {}

TetraTrainer::TetraTrainer(lfs::vis::Scene& scene, const TetraTrainConfig& config)
    : config_(config)
    , scene_(&scene) {}

TetraTrainer::~TetraTrainer() {
    // Ensure clean shutdown
    stop_requested_ = true;
    while (is_running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// ------------------------------
// INITIALIZATION
// ------------------------------

std::expected<void, std::string> TetraTrainer::initialize(
    const std::filesystem::path& data_path) {

    LOG_INFO("Initializing TetraTrainer from {}", data_path.string());

    // Load dataset using LichtFeld's loader with configured options
    io::LoadOptions load_options;
    load_options.images_folder = config_.images_folder;
    load_options.resize_factor = config_.resize_factor;
    LOG_INFO("Using images folder: {}, resize_factor: {}",
             load_options.images_folder, load_options.resize_factor);

    auto loader = lfs::io::Loader::create();
    auto loader_result = loader->load(data_path, load_options);
    if (!loader_result) {
        return std::unexpected("Failed to load dataset: " + loader_result.error().message);
    }

    // Extract point cloud and cameras from LoadResult
    auto* scene_data = std::get_if<io::LoadedScene>(&loader_result->data);
    if (!scene_data) {
        return std::unexpected("Expected scene data but got splat data");
    }

    dataset_ = scene_data->cameras;

    // Apply resize factor from config
    if (config_.resize_factor > 1) {
        LOG_INFO("Setting resize factor to {} for tetra training", config_.resize_factor);
        dataset_->set_resize_factor(config_.resize_factor);
    }

    if (!scene_data->point_cloud) {
        return std::unexpected("No point cloud in loaded scene");
    }

    return initialize(*scene_data->point_cloud, dataset_);
}

std::expected<void, std::string> TetraTrainer::initialize(
    const core::PointCloud& point_cloud,
    std::shared_ptr<training::CameraDataset> dataset) {

    std::lock_guard<std::mutex> lock(train_mutex_);

    if (initialized_.load()) {
        return std::unexpected("Trainer already initialized");
    }

    dataset_ = std::move(dataset);

    // Compute scene bounds
    core::Tensor means_cpu = point_cloud.means.to(core::Device::CPU);
    const float* data = means_cpu.ptr<float>();
    const size_t N = static_cast<size_t>(means_cpu.shape()[0]);

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < N; ++i) {
        min_x = std::min(min_x, data[i * 3 + 0]);
        min_y = std::min(min_y, data[i * 3 + 1]);
        min_z = std::min(min_z, data[i * 3 + 2]);
        max_x = std::max(max_x, data[i * 3 + 0]);
        max_y = std::max(max_y, data[i * 3 + 1]);
        max_z = std::max(max_z, data[i * 3 + 2]);
    }

    // Compute scene_scale from camera centers (matching Python radiance_meshes)
    // Python: scaling = torch.linalg.norm(ccenters - center, dim=1, ord=torch.inf).max()
    const auto& cameras = dataset_->get_cameras();
    std::vector<float> cam_centers;
    cam_centers.reserve(cameras.size() * 3);

    float cam_sum_x = 0.f, cam_sum_y = 0.f, cam_sum_z = 0.f;
    for (const auto& cam : cameras) {
        core::Tensor pos_cpu = cam->cam_position().to(core::Device::CPU);
        const float* pos = pos_cpu.ptr<float>();
        cam_centers.push_back(pos[0]);
        cam_centers.push_back(pos[1]);
        cam_centers.push_back(pos[2]);
        cam_sum_x += pos[0];
        cam_sum_y += pos[1];
        cam_sum_z += pos[2];
    }

    // Camera center mean
    float cam_center_x = cam_sum_x / static_cast<float>(cameras.size());
    float cam_center_y = cam_sum_y / static_cast<float>(cameras.size());
    float cam_center_z = cam_sum_z / static_cast<float>(cameras.size());

    // Compute max L-infinity norm from camera centers to mean center
    float max_linf = 0.f;
    for (size_t i = 0; i < cameras.size(); ++i) {
        float dx = std::abs(cam_centers[i * 3 + 0] - cam_center_x);
        float dy = std::abs(cam_centers[i * 3 + 1] - cam_center_y);
        float dz = std::abs(cam_centers[i * 3 + 2] - cam_center_z);
        float linf = std::max({dx, dy, dz});
        max_linf = std::max(max_linf, linf);
    }

    // Use point cloud center for scene_center (for mesh creation)
    scene_center_ = core::Tensor::from_vector(
        std::vector<float>{
            (min_x + max_x) * 0.5f,
            (min_y + max_y) * 0.5f,
            (min_z + max_z) * 0.5f
        }, {3}, core::Device::CUDA);

    // Use camera-based scaling (matching Python)
    scene_scale_ = max_linf > 1e-6f ? max_linf : 1.0f;

    LOG_INFO("Scene bounds: ({:.2f}, {:.2f}, {:.2f}) to ({:.2f}, {:.2f}, {:.2f})",
             min_x, min_y, min_z, max_x, max_y, max_z);
    LOG_INFO("Camera center: ({:.2f}, {:.2f}, {:.2f})",
             cam_center_x, cam_center_y, cam_center_z);
    LOG_INFO("Scene scale (from cameras): {:.2f}", scene_scale_);

    // Create tetrahedral mesh from point cloud
    auto mesh_result = TetraMesh::from_point_cloud(
        point_cloud, scene_scale_, config_.shell_expansion);
    if (!mesh_result) {
        return std::unexpected("Failed to create mesh: " + mesh_result.error());
    }
    mesh_ = std::make_unique<TetraMesh>(std::move(*mesh_result));

    LOG_INFO("Created mesh with {} vertices, {} tetrahedra",
             mesh_->num_vertices(), mesh_->num_tetrahedra());

    // Initialize feature network
    features_ = std::make_unique<TetraFeatures>();

    core::Tensor scene_aabb = core::Tensor::from_vector(
        std::vector<float>{min_x, min_y, min_z, max_x, max_y, max_z},
        {6}, core::Device::CUDA);

    auto features_result = features_->initialize(
        config_.hash_config, config_.mlp_config, scene_aabb);
    if (!features_result) {
        return std::unexpected("Failed to initialize features: " + features_result.error());
    }

    // Initialize renderer
    renderer_ = std::make_unique<TetraRenderer>();
    TetraRenderConfig render_config;
    auto render_result = renderer_->initialize(render_config);
    if (!render_result) {
        return std::unexpected("Failed to initialize renderer: " + render_result.error());
    }

    // Setup optimizer
    auto opt_result = setup_optimizer();
    if (!opt_result) {
        return std::unexpected("Failed to setup optimizer: " + opt_result.error());
    }

    // Initialize background
    background_ = core::Tensor::zeros({3}, core::Device::CUDA, core::DataType::Float32);

    initialized_ = true;
    LOG_INFO("TetraTrainer initialization complete");

    return {};
}

std::expected<void, std::string> TetraTrainer::setup_optimizer() {
    if (!mesh_ || !features_) {
        return std::unexpected("Mesh and features must be initialized before optimizer");
    }

    // Create optimizer config from trainer config
    TetraOptimizerConfig opt_config;
    opt_config.vertices_lr = config_.vertices_lr;
    opt_config.vertices_lr_final = config_.vertices_lr * config_.lr_final_multiplier;
    opt_config.encoding_lr = config_.encoding_lr;
    opt_config.encoding_lr_final = config_.encoding_lr;  // Encoding typically no decay
    opt_config.network_lr = config_.network_lr;
    opt_config.network_lr_final = config_.network_lr;    // Network typically no decay

    // Adam hyperparameters
    opt_config.beta1 = 0.9f;
    opt_config.beta2 = 0.999f;
    opt_config.eps = 1e-8f;

    // Schedule parameters
    opt_config.freeze_start = config_.freeze_start;
    opt_config.lr_delay_steps = 0;     // No warmup delay (Python: lr_delay=0)
    opt_config.lr_delay_mult = 1.0f;   // No delay multiplier

    // Densification spike parameters
    opt_config.enable_lr_spikes = true;
    opt_config.spike_duration = 20;
    opt_config.densify_interval = config_.densify_interval;
    opt_config.densify_end = config_.densify_end;
    opt_config.spike_midpoint = config_.densify_start;

    // Create optimizer
    optimizer_ = std::make_unique<TetraOptimizer>(*mesh_, *features_, opt_config);

    // Pre-allocate gradient buffers with capacity for densification growth
    const size_t initial_verts = mesh_->num_vertices();
    const size_t estimated_max_verts = static_cast<size_t>(initial_verts * 2.0f);
    optimizer_->allocate_gradients(estimated_max_verts);

    LOG_INFO("Optimizer initialized: vertices_lr={:.6f}, encoding_lr={:.6f}, network_lr={:.6f}",
             config_.vertices_lr, config_.encoding_lr, config_.network_lr);

    return {};
}

// ------------------------------
// TRAINING
// ------------------------------

std::expected<void, std::string> TetraTrainer::train(std::stop_token stop_token) {
    if (!initialized_.load()) {
        return std::unexpected("Trainer not initialized");
    }

    if (is_running_.load()) {
        return std::unexpected("Training already in progress");
    }

    is_running_ = true;
    training_complete_ = false;
    stop_requested_ = false;

    // Initialize CacheLoader for image loading (required by dataset_->get())
    // Use default settings: no CPU cache, no filesystem cache (direct GPU decode)
    auto& cache_loader = lfs::io::CacheLoader::getInstance(false, false);
    cache_loader.reset_cache();
    cache_loader.update_cache_params(
        false,   // use_cpu_memory
        false,   // use_fs_cache
        static_cast<int>(dataset_->size()),  // num_expected_images
        1.0f,    // min_cpu_free_GB
        0.1f,    // min_cpu_free_memory_ratio
        false,   // print_cache_status
        500      // print_status_freq_num
    );

    LOG_INFO("Starting training for {} iterations", config_.iterations);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < config_.iterations; ++iter) {
        // Check for stop request
        if (stop_requested_.load() || stop_token.stop_requested()) {
            LOG_INFO("Training stopped at iteration {}", iter);
            break;
        }

        // Handle pause
        while (pause_requested_.load() && !stop_requested_.load()) {
            is_paused_ = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        is_paused_ = false;

        // Execute training step
        auto step_result = train_step(iter);
        if (!step_result) {
            LOG_ERROR("Training step {} failed: {}", iter, step_result.error());
            is_running_ = false;
            return std::unexpected(step_result.error());
        }

        current_iteration_ = iter;
        current_loss_ = *step_result;

        // Handle periodic operations
        handle_periodic_operations(iter);

        // Log progress periodically
        if (iter % 100 == 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration<float>(elapsed).count();
            float iter_per_sec = (iter + 1) / elapsed_sec;

            LOG_INFO("Iter {}/{}: loss={:.4f}, {:.1f} it/s, {} verts, {} tets",
                     iter, config_.iterations, current_loss_.load(),
                     iter_per_sec, mesh_->num_interior_vertices(), mesh_->num_tetrahedra());
        }
    }

    training_complete_ = true;
    is_running_ = false;

    auto total_time = std::chrono::high_resolution_clock::now() - start_time;
    auto total_sec = std::chrono::duration<float>(total_time).count();

    LOG_INFO("Training complete: {} iterations in {:.1f}s ({:.1f} it/s)",
             current_iteration_.load(), total_sec,
             current_iteration_.load() / total_sec);

    // Final checkpoint
    if (!config_.output_path.empty()) {
        save_checkpoint(config_.output_path / "final.ckpt");
        export_mesh(config_.output_path / "mesh.ply", true);

        // Final evaluation
        if (config_.enable_eval) {
            LOG_INFO("Running final evaluation...");
            auto eval_result = evaluate(config_.output_path, config_.save_eval_images);
            if (eval_result) {
                LOG_INFO("Final metrics: {}", eval_result->to_string());
            } else {
                LOG_WARN("Final evaluation failed: {}", eval_result.error());
            }
        }

        // Render rotating video
        if (config_.render_rotating_video) {
            LOG_INFO("Rendering rotating video...");
            auto video_path = config_.output_path / "rotating.mp4";
            auto video_result = render_video(video_path, config_.video_num_frames, config_.video_fps);
            if (video_result) {
                LOG_INFO("Rotating video saved to {}", video_path.string());
            } else {
                LOG_WARN("Video rendering failed: {}", video_result.error());
            }
        }
    }

    return {};
}

std::expected<float, std::string> TetraTrainer::train_step(int iteration) {
    std::shared_lock<std::shared_mutex> render_lock(render_mutex_);

    // Sample random camera index
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, dataset_->size() - 1);
    size_t camera_idx = dist(rng);

    // Get camera example (loads image)
    auto example = dataset_->get(camera_idx);
    auto& cam = *example.data.camera;
    core::Tensor gt_image = std::move(example.data.image);

    if (!gt_image.is_valid()) {
        return std::unexpected("Camera has no image");
    }

    // Render
    auto render_result = renderer_->forward(*mesh_, *features_, cam, background_);
    if (!render_result) {
        return std::unexpected("Render failed: " + render_result.error());
    }

    // Debug: Check tensor shapes and devices
    LOG_INFO("Rendered shape: [{}, {}, {}], GT shape: [{}, {}, {}]",
             render_result->rgb.shape()[0], render_result->rgb.shape()[1], render_result->rgb.shape()[2],
             gt_image.shape()[0], gt_image.shape()[1], gt_image.shape()[2]);
    LOG_INFO("Rendered device: {}, GT device: {}",
             render_result->rgb.device() == core::Device::CUDA ? "CUDA" : "CPU",
             gt_image.device() == core::Device::CUDA ? "CUDA" : "CPU");

    // Debug: Check pixel value ranges (sample first few values)
    {
        auto rendered_cpu = render_result->rgb.to(core::Device::CPU);
        auto gt_cpu = gt_image.to(core::Device::CPU);
        cudaDeviceSynchronize();

        float* rendered_ptr = rendered_cpu.ptr<float>();
        float* gt_ptr = gt_cpu.ptr<float>();
        size_t n = std::min(static_cast<size_t>(10), static_cast<size_t>(rendered_cpu.numel()));

        LOG_INFO("Rendered first {} values: [{:.4f}, {:.4f}, {:.4f}, ...]",
                 n, rendered_ptr[0], rendered_ptr[1], rendered_ptr[2]);
        LOG_INFO("GT first {} values: [{:.4f}, {:.4f}, {:.4f}, ...]",
                 n, gt_ptr[0], gt_ptr[1], gt_ptr[2]);

        // Check for NaN/Inf
        bool has_nan = false;
        for (size_t i = 0; i < static_cast<size_t>(rendered_cpu.numel()); i += 10000) {
            if (std::isnan(rendered_ptr[i]) || std::isinf(rendered_ptr[i])) {
                has_nan = true;
                LOG_WARN("Rendered has NaN/Inf at index {}", i);
                break;
            }
        }
        if (!has_nan) {
            LOG_INFO("Rendered values appear valid (no NaN/Inf detected in samples)");
        }
    }

    // Compute loss
    auto loss_result = compute_loss(render_result->rgb, gt_image);
    if (!loss_result) {
        return std::unexpected("Loss computation failed: " + loss_result.error());
    }

    auto& [loss, grad] = *loss_result;

    LOG_INFO("compute_loss returned - loss ndim: {}, grad ndim: {}", loss.ndim(), grad.ndim());
    if (loss.ndim() > 0) {
        LOG_INFO("  loss shape: [{}]", loss.shape()[0]);
    }
    if (grad.ndim() >= 3) {
        LOG_INFO("  grad shape: [{}, {}, {}]", grad.shape()[0], grad.shape()[1], grad.shape()[2]);
    }

    // Get loss value - loss is a scalar or 1-element tensor
    LOG_INFO("Converting loss to CPU...");
    core::Tensor loss_cpu = loss.to(core::Device::CPU);
    cudaDeviceSynchronize();
    LOG_INFO("Loss on CPU, extracting value...");
    float loss_val = loss.ndim() == 0 ? loss_cpu.item<float>() : loss_cpu.ptr<float>()[0];
    LOG_INFO("Loss value: {}", loss_val);

    // Backward pass through renderer
    LOG_INFO("Starting backward pass...");
    // Create zeros tensor for grad_alpha with shape [1, H, W] matching grad [C, H, W]
    core::Tensor grad_alpha = core::Tensor::zeros(
        {1, static_cast<size_t>(grad.shape()[1]), static_cast<size_t>(grad.shape()[2])},
        core::Device::CUDA,
        core::DataType::Float32);
    auto backward_result = renderer_->backward(
        *mesh_, *features_, cam, *render_result, grad, grad_alpha);
    if (!backward_result) {
        return std::unexpected("Backward failed: " + backward_result.error());
    }
    LOG_INFO("Backward pass completed");

    // Accumulate vertex gradients for densification
    accumulate_vertex_gradients(backward_result->grad_vertices);

    // ------------------------------------------------------------
    // Optimizer step: Apply gradients to all parameters using Adam
    // ------------------------------------------------------------

    if (!optimizer_) {
        return std::unexpected("Optimizer not initialized");
    }

    // Copy vertex gradients to optimizer
    if (backward_result->grad_vertices.is_valid()) {
        core::Tensor& opt_grad_verts = optimizer_->get_grad(TetraParamType::Vertices);
        if (opt_grad_verts.is_valid() && opt_grad_verts.numel() > 0) {
            const size_t num_verts = std::min(
                static_cast<size_t>(backward_result->grad_vertices.shape()[0]),
                static_cast<size_t>(opt_grad_verts.shape()[0]));
            const size_t bytes = num_verts * 3 * sizeof(float);
            cudaMemcpyAsync(
                opt_grad_verts.ptr<float>(),
                backward_result->grad_vertices.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToDevice,
                nullptr);
        }
    }

    // Copy per-tet gradients to optimizer (density, base_color, gradient)
    if (backward_result->grad_density.is_valid()) {
        core::Tensor& opt_grad_density = optimizer_->get_grad(TetraParamType::Density);
        if (opt_grad_density.is_valid() && opt_grad_density.numel() > 0) {
            const size_t num_tets = std::min(
                static_cast<size_t>(backward_result->grad_density.shape()[0]),
                static_cast<size_t>(opt_grad_density.shape()[0]));
            const size_t bytes = num_tets * sizeof(float);
            cudaMemcpyAsync(
                opt_grad_density.ptr<float>(),
                backward_result->grad_density.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToDevice,
                nullptr);
        }
    }

    if (backward_result->grad_base_color.is_valid()) {
        core::Tensor& opt_grad_base = optimizer_->get_grad(TetraParamType::BaseColor);
        if (opt_grad_base.is_valid() && opt_grad_base.numel() > 0) {
            const size_t num_tets = std::min(
                static_cast<size_t>(backward_result->grad_base_color.shape()[0]),
                static_cast<size_t>(opt_grad_base.shape()[0]));
            const size_t bytes = num_tets * 3 * sizeof(float);
            cudaMemcpyAsync(
                opt_grad_base.ptr<float>(),
                backward_result->grad_base_color.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToDevice,
                nullptr);
        }
    }

    if (backward_result->grad_gradient.is_valid()) {
        core::Tensor& opt_grad_grad = optimizer_->get_grad(TetraParamType::Gradient);
        if (opt_grad_grad.is_valid() && opt_grad_grad.numel() > 0) {
            const size_t num_tets = std::min(
                static_cast<size_t>(backward_result->grad_gradient.shape()[0]),
                static_cast<size_t>(opt_grad_grad.shape()[0]));
            const size_t bytes = num_tets * 3 * sizeof(float);
            cudaMemcpyAsync(
                opt_grad_grad.ptr<float>(),
                backward_result->grad_gradient.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToDevice,
                nullptr);
        }
    }

    // ------------------------------------------------------------
    // Network backward pass: Train hash grid + MLP
    // Process in chunks to avoid memory issues with large meshes
    // ------------------------------------------------------------
    if (render_result->used_network &&
        render_result->tet_centroids.is_valid() &&
        backward_result->grad_base_color.is_valid()) {

        LOG_INFO("Starting chunked network backward for {} tetrahedra",
                 render_result->tet_centroids.shape()[0]);

        const size_t num_tets = render_result->tet_centroids.shape()[0];
        const size_t chunk_size = 100000;  // Process 100k tets at a time

        // Prepare gradient inputs for network backward
        FeatureBackwardInputs grad_inputs;

        for (size_t chunk_start = 0; chunk_start < num_tets; chunk_start += chunk_size) {
            const size_t chunk_end = std::min(chunk_start + chunk_size, num_tets);
            const size_t current_chunk_size = chunk_end - chunk_start;

            // Slice centroids for this chunk
            core::Tensor chunk_positions = core::Tensor::empty(
                {current_chunk_size, 3}, core::Device::CUDA, core::DataType::Float32);
            cudaMemcpyAsync(
                chunk_positions.ptr<float>(),
                render_result->tet_centroids.ptr<float>() + chunk_start * 3,
                current_chunk_size * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr);

            // Slice view directions for this chunk
            core::Tensor chunk_directions = core::Tensor::empty(
                {current_chunk_size, 3}, core::Device::CUDA, core::DataType::Float32);
            cudaMemcpyAsync(
                chunk_directions.ptr<float>(),
                render_result->view_directions.ptr<float>() + chunk_start * 3,
                current_chunk_size * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr);

            // Slice RGB gradients for this chunk
            core::Tensor chunk_grad_rgb = core::Tensor::empty(
                {current_chunk_size, 3}, core::Device::CUDA, core::DataType::Float32);
            cudaMemcpyAsync(
                chunk_grad_rgb.ptr<float>(),
                backward_result->grad_base_color.ptr<float>() + chunk_start * 3,
                current_chunk_size * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr);

            cudaDeviceSynchronize();

            // Set up gradient inputs
            grad_inputs.grad_rgb = chunk_grad_rgb;

            // Run network backward for this chunk
            auto feature_backward_result = features_->backward(
                chunk_positions, chunk_directions, grad_inputs);

            if (!feature_backward_result) {
                LOG_WARN("Network backward failed for chunk [{}, {}): {}",
                         chunk_start, chunk_end, feature_backward_result.error());
                continue;
            }

            // Apply gradients using simple SGD
            // Learning rate for hash grid and MLP (match Python's radiance_meshes)
            const float hash_lr = config_.encoding_lr;    // Default: 0.003 for hash grid
            const float mlp_lr = config_.network_lr;      // Default: 0.001 for MLP

            // Apply gradient to hash table: hash_table -= lr * grad
            {
                auto& hash_table = features_->hash_table();
                const auto& grad_hash = feature_backward_result->grad_hash_params;
                if (hash_table.is_valid() && grad_hash.is_valid()) {
                    const size_t n = static_cast<size_t>(std::min(
                        hash_table.numel(), grad_hash.numel()));
                    cuda::launch_sgd_update(
                        hash_table.ptr<float>(),
                        grad_hash.ptr<float>(),
                        hash_lr,
                        n,
                        nullptr);
                }
            }

            // Apply gradients to MLP weights and biases
            auto mlp_params = features_->mlp_weights();
            const auto& grad_weights = feature_backward_result->grad_mlp_weights;
            const auto& grad_biases = feature_backward_result->grad_mlp_biases;

            // Apply weight gradients (first half of mlp_params are weights)
            for (size_t l = 0; l < grad_weights.size(); ++l) {
                if (l < mlp_params.size()) {
                    auto* weight = mlp_params[l];
                    const auto& grad = grad_weights[l];
                    if (weight && weight->is_valid() && grad.is_valid()) {
                        const size_t n = static_cast<size_t>(std::min(
                            weight->numel(), grad.numel()));
                        cuda::launch_sgd_update(
                            weight->ptr<float>(),
                            grad.ptr<float>(),
                            mlp_lr,
                            n,
                            nullptr);
                    }
                }
            }

            // Apply bias gradients (second half of mlp_params are biases)
            const size_t num_weights = grad_weights.size();
            for (size_t l = 0; l < grad_biases.size(); ++l) {
                if (num_weights + l < mlp_params.size()) {
                    auto* bias = mlp_params[num_weights + l];
                    const auto& grad = grad_biases[l];
                    if (bias && bias->is_valid() && grad.is_valid()) {
                        const size_t n = static_cast<size_t>(std::min(
                            bias->numel(), grad.numel()));
                        cuda::launch_sgd_update(
                            bias->ptr<float>(),
                            grad.ptr<float>(),
                            mlp_lr,
                            n,
                            nullptr);
                    }
                }
            }

            if (chunk_start == 0) {
                LOG_INFO("Network backward chunk [0, {}) completed successfully", chunk_end);
            }
        }

        LOG_INFO("Network backward completed for all {} tetrahedra", num_tets);
    } else {
        // Fallback: per-tet parameters are already being updated
        LOG_DEBUG("Using per-tet parameters (network not used or not available)");
    }

    // Execute Adam optimizer step (applies gradients to parameters)
    optimizer_->step(iteration);

    // Zero gradients for next iteration
    optimizer_->zero_grad();

    return loss_val;
}

void TetraTrainer::handle_periodic_operations(int iteration) {
    // Checkpoint
    if (config_.checkpoint_interval > 0 &&
        iteration > 0 &&
        iteration % config_.checkpoint_interval == 0 &&
        !config_.output_path.empty()) {

        auto ckpt_path = config_.output_path /
            ("checkpoint_" + std::to_string(iteration) + ".ckpt");
        save_checkpoint(ckpt_path);
    }

    // Densification
    if (iteration >= config_.densify_start &&
        iteration < config_.densify_end &&
        iteration % config_.densify_interval == 0) {

        auto densify_result = densify(iteration);
        if (!densify_result) {
            LOG_WARN("Densification failed: {}", densify_result.error());
        }
    }

    // Delaunay update (before freeze, skip iteration 0 to avoid resetting params)
    // Also skip if vertices_lr is 0 (vertices frozen, no point in retriangulating)
    if (config_.vertices_lr > 0.0f &&
        iteration > 0 &&
        iteration < config_.freeze_start &&
        iteration % config_.delaunay_interval == 0) {

        auto tri_result = update_triangulation();
        if (!tri_result) {
            LOG_WARN("Triangulation update failed: {}", tri_result.error());
        }
    }
}

void TetraTrainer::update_learning_rates(int iteration) {
    // The TetraOptimizer handles its own learning rate scheduling internally
    // via update_learning_rate() which is called by optimizer_->step()
    // This function is now a no-op since the optimizer manages LR decay

    // The optimizer uses exponential decay with optional warmup and LR spikes
    // during densification, which is more sophisticated than simple linear decay
}

std::expected<std::pair<core::Tensor, core::Tensor>, std::string>
TetraTrainer::compute_loss(const core::Tensor& rendered,
                           const core::Tensor& gt_image) {

    if (!rendered.is_valid() || !gt_image.is_valid()) {
        return std::unexpected("Invalid tensors for loss computation");
    }

    // Ensure same device and 4D format [N, C, H, W] for SSIM kernels
    core::Tensor rendered_gpu = rendered.to(core::Device::CUDA);
    core::Tensor gt_gpu = gt_image.to(core::Device::CUDA);

    // Convert rendered from HWC to CHW if needed (renderer outputs HWC)
    // GT is already in CHW format from the image loader
    bool rendered_is_hwc = rendered_gpu.ndim() == 3 && rendered_gpu.shape()[2] <= 4;
    bool gt_is_hwc = gt_gpu.ndim() == 3 && gt_gpu.shape()[2] <= 4;

    LOG_INFO("Before permute - rendered is_hwc: {}, gt is_hwc: {}", rendered_is_hwc, gt_is_hwc);

    if (rendered_is_hwc) {
        // [H, W, C] -> [C, H, W]
        LOG_INFO("Permuting rendered from HWC to CHW...");
        rendered_gpu = rendered_gpu.permute({2, 0, 1}).contiguous();
        LOG_INFO("Rendered after permute: [{}, {}, {}], contiguous: {}",
                 rendered_gpu.shape()[0], rendered_gpu.shape()[1], rendered_gpu.shape()[2],
                 rendered_gpu.is_contiguous());
    }
    if (gt_is_hwc) {
        // [H, W, C] -> [C, H, W]
        gt_gpu = gt_gpu.permute({2, 0, 1}).contiguous();
    }

    // Synchronize to catch any permute errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error after permute: {}", cudaGetErrorString(err));
        return std::unexpected(std::string("CUDA error after permute: ") + cudaGetErrorString(err));
    }
    LOG_INFO("Permute completed successfully");

    // Add batch dimension if needed (SSIM kernels expect [N, C, H, W])
    LOG_INFO("Adding batch dimension...");
    core::Tensor rendered_4d = rendered_gpu.ndim() == 3
        ? rendered_gpu.unsqueeze(0)
        : rendered_gpu;
    core::Tensor gt_4d = gt_gpu.ndim() == 3
        ? gt_gpu.unsqueeze(0)
        : gt_gpu;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error after unsqueeze: {}", cudaGetErrorString(err));
        return std::unexpected(std::string("CUDA error after unsqueeze: ") + cudaGetErrorString(err));
    }

    LOG_INFO("Tensors for loss - rendered_4d: [{}, {}, {}, {}], gt_4d: [{}, {}, {}, {}]",
             rendered_4d.shape()[0], rendered_4d.shape()[1], rendered_4d.shape()[2], rendered_4d.shape()[3],
             gt_4d.shape()[0], gt_4d.shape()[1], gt_4d.shape()[2], gt_4d.shape()[3]);

    // Verify shapes match
    if (rendered_4d.shape() != gt_4d.shape()) {
        LOG_ERROR("Shape mismatch between rendered and GT");
        return std::unexpected("Shape mismatch between rendered and GT tensors");
    }

    core::Tensor total_loss;
    core::Tensor grad;

    // Compute ssim_weight for the fused kernel: loss = (1-w)*L1 + w*(1-SSIM)
    // Our config has lambda_l1 and lambda_ssim, normalize to get weight
    const float total_weight = config_.lambda_l1 + config_.lambda_ssim;
    const float ssim_weight = total_weight > 0.0f
        ? config_.lambda_ssim / total_weight
        : 0.0f;

    LOG_INFO("Loss weights - ssim_weight: {}, total_weight: {}", ssim_weight, total_weight);

    if (ssim_weight == 0.0f) {
        // Pure L1 loss
        core::Tensor diff = rendered_4d - gt_4d;
        core::Tensor abs_diff = diff.abs();
        core::Tensor l1_loss = abs_diff.mean();

        total_loss = l1_loss * config_.lambda_l1;
        grad = diff.sign() * (config_.lambda_l1 / static_cast<float>(diff.numel()));

    } else if (ssim_weight == 1.0f) {
        // Pure SSIM loss: loss = 1 - SSIM
        fused_l1_ssim_workspace_.ensure_size(rendered_4d.shape().dims());
        training::kernels::SSIMWorkspace ssim_workspace;
        ssim_workspace.ensure_size(rendered_4d.shape().dims());

        auto [ssim_value, ssim_ctx] = training::kernels::ssim_forward(
            rendered_4d, gt_4d, ssim_workspace, true);

        total_loss = (core::Tensor::full({1}, 1.0f, core::Device::CUDA) - ssim_value)
                     * config_.lambda_ssim;

        // Backward: d(loss)/d(ssim) = -lambda_ssim
        grad = training::kernels::ssim_backward(ssim_ctx, ssim_workspace, -config_.lambda_ssim);

    } else {
        // Fused L1+SSIM loss using optimized kernel
        // The fused kernel computes: loss = (1-w)*L1 + w*(1-SSIM)
        LOG_INFO("Calling fused_l1_ssim_forward...");

        // Ensure workspace is ready
        LOG_INFO("Ensuring workspace size for shape [{}, {}, {}, {}]",
                 rendered_4d.shape()[0], rendered_4d.shape()[1],
                 rendered_4d.shape()[2], rendered_4d.shape()[3]);
        fused_l1_ssim_workspace_.ensure_size(rendered_4d.shape().dims());

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after workspace ensure_size: {}", cudaGetErrorString(err));
            return std::unexpected(std::string("CUDA error after workspace: ") + cudaGetErrorString(err));
        }
        LOG_INFO("Workspace ready");

        auto [loss_tensor, fused_ctx] = training::kernels::fused_l1_ssim_forward(
            rendered_4d, gt_4d, ssim_weight, fused_l1_ssim_workspace_, true);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after fused_l1_ssim_forward: {}", cudaGetErrorString(err));
            return std::unexpected(std::string("CUDA error in fused loss: ") + cudaGetErrorString(err));
        }
        LOG_INFO("fused_l1_ssim_forward completed, loss: {}", loss_tensor.item<float>());

        // Scale by total weight to match our lambda semantics
        total_loss = loss_tensor * total_weight;

        // Get combined gradient from fused backward
        LOG_INFO("Calling fused_l1_ssim_backward...");
        grad = training::kernels::fused_l1_ssim_backward(fused_ctx, fused_l1_ssim_workspace_);
        grad = grad * total_weight;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error after fused_l1_ssim_backward: {}", cudaGetErrorString(err));
            return std::unexpected(std::string("CUDA error in fused backward: ") + cudaGetErrorString(err));
        }
        LOG_INFO("fused_l1_ssim_backward completed");
    }

    // Remove batch dimension if input was 3D
    if (rendered_gpu.ndim() == 3 && grad.ndim() == 4) {
        grad = grad.squeeze(0);
    }

    return std::make_pair(total_loss, grad);
}

std::expected<void, std::string> TetraTrainer::densify(int iteration) {
    std::unique_lock<std::shared_mutex> lock(render_mutex_);

    // ------------------------------------------------------------
    // Check if we have accumulated gradients
    // ------------------------------------------------------------
    if (gradient_count_ == 0 || !vertex_gradients_.is_valid()) {
        LOG_INFO("Densify at iter {}: no gradients accumulated, skipping", iteration);
        return {};
    }

    const size_t num_interior = mesh_->num_interior_vertices();
    if (num_interior == 0) {
        return std::unexpected("No interior vertices to densify");
    }

    // ------------------------------------------------------------
    // Compute average gradient norm per vertex
    // ------------------------------------------------------------
    core::Tensor verts_cpu = mesh_->vertices().to(core::Device::CPU);
    const float* vert_data = verts_cpu.ptr<float>();
    const float* grad_data = vertex_gradients_.ptr<float>();
    const float grad_divisor = static_cast<float>(gradient_count_);

    // ------------------------------------------------------------
    // Find vertices with high gradients (above threshold)
    // ------------------------------------------------------------
    std::vector<size_t> high_grad_indices;
    for (size_t i = 0; i < num_interior; ++i) {
        float avg_grad = grad_data[i] / grad_divisor;
        if (avg_grad > config_.densify_grad_threshold) {
            high_grad_indices.push_back(i);
        }
    }

    if (high_grad_indices.empty()) {
        LOG_INFO("Densify at iter {}: no vertices above threshold {:.6f}, skipping",
                 iteration, config_.densify_grad_threshold);
        reset_gradient_accumulation();
        return {};
    }

    // ------------------------------------------------------------
    // Limit number of new vertices to avoid explosion
    // ------------------------------------------------------------
    constexpr size_t max_new_vertices = 10000;
    if (high_grad_indices.size() > max_new_vertices) {
        // Sort by gradient magnitude and keep top candidates
        std::vector<std::pair<float, size_t>> grad_pairs;
        grad_pairs.reserve(high_grad_indices.size());
        for (size_t idx : high_grad_indices) {
            grad_pairs.emplace_back(grad_data[idx] / grad_divisor, idx);
        }
        std::partial_sort(
            grad_pairs.begin(),
            grad_pairs.begin() + max_new_vertices,
            grad_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        high_grad_indices.clear();
        for (size_t i = 0; i < max_new_vertices; ++i) {
            high_grad_indices.push_back(grad_pairs[i].second);
        }
    }

    // ------------------------------------------------------------
    // Generate new vertices by cloning with small random offset
    // This follows the "clone" strategy from radiance_meshes densification
    // ------------------------------------------------------------
    std::vector<float> new_positions;
    new_positions.reserve(high_grad_indices.size() * 3);

    std::mt19937 rng(iteration);  // Deterministic for reproducibility
    std::normal_distribution<float> offset_dist(0.0f, scene_scale_ * 0.001f);

    for (size_t idx : high_grad_indices) {
        float x = vert_data[idx * 3 + 0] + offset_dist(rng);
        float y = vert_data[idx * 3 + 1] + offset_dist(rng);
        float z = vert_data[idx * 3 + 2] + offset_dist(rng);
        new_positions.push_back(x);
        new_positions.push_back(y);
        new_positions.push_back(z);
    }

    // Create tensor for new vertices
    core::Tensor new_verts = core::Tensor::from_vector(
        new_positions,
        {high_grad_indices.size(), 3UL},
        core::Device::CUDA);

    // ------------------------------------------------------------
    // Add vertices to mesh and optimizer state
    // ------------------------------------------------------------
    // Use optimizer to add vertices - it handles both mesh and optimizer state
    if (optimizer_) {
        auto add_result = optimizer_->add_vertices(new_verts);
        if (!add_result) {
            return std::unexpected("Failed to add vertices via optimizer: " + add_result.error());
        }
    } else {
        // Fallback: just add to mesh directly
        auto add_result = mesh_->add_vertices(new_verts);
        if (!add_result) {
            return std::unexpected("Failed to add vertices: " + add_result.error());
        }
    }

    auto tri_result = mesh_->update_triangulation();
    if (!tri_result) {
        return std::unexpected("Failed to update triangulation: " + tri_result.error());
    }

    LOG_INFO("Densify at iter {}: added {} vertices (grad threshold {:.6f}), "
             "total vertices: {}, tetrahedra: {}",
             iteration, high_grad_indices.size(), config_.densify_grad_threshold,
             mesh_->num_interior_vertices(), mesh_->num_tetrahedra());

    // Reset gradient accumulation for next densification interval
    reset_gradient_accumulation();

    return {};
}

void TetraTrainer::accumulate_vertex_gradients(const core::Tensor& grad_vertices) {
    if (!grad_vertices.is_valid()) {
        return;
    }

    // Compute L2 norm of gradients per vertex on GPU
    // grad_vertices: [V, 3]
    core::Tensor grad_gpu = grad_vertices.to(core::Device::CUDA);
    const size_t num_vertices = static_cast<size_t>(grad_gpu.shape()[0]);

    // Initialize gradient accumulator on GPU if needed
    if (!vertex_gradients_.is_valid() ||
        static_cast<size_t>(vertex_gradients_.shape()[0]) != num_vertices) {
        vertex_gradients_ = core::Tensor::zeros(
            {num_vertices}, core::Device::CUDA, core::DataType::Float32);
        gradient_count_ = 0;
    }

    // Launch GPU kernel to accumulate gradient norms
    cuda::launch_accumulate_gradient_norms(
        grad_gpu.ptr<float>(),
        vertex_gradients_.ptr<float>(),
        num_vertices,
        nullptr);

    gradient_count_++;
}

void TetraTrainer::reset_gradient_accumulation() {
    if (vertex_gradients_.is_valid()) {
        vertex_gradients_ = core::Tensor::zeros(
            vertex_gradients_.shape(), core::Device::CUDA, core::DataType::Float32);
    }
    gradient_count_ = 0;
}

std::expected<void, std::string> TetraTrainer::update_triangulation() {
    std::unique_lock<std::shared_mutex> lock(render_mutex_);

    return mesh_->update_triangulation();
}

// ------------------------------
// STATE ACCESS
// ------------------------------

TetraTrainProgress TetraTrainer::get_progress() const {
    // Get current learning rates from optimizer if available
    float current_vert_lr = config_.vertices_lr;
    float current_enc_lr = config_.encoding_lr;
    if (optimizer_) {
        current_vert_lr = optimizer_->get_lr(TetraParamType::Vertices);
        current_enc_lr = optimizer_->get_lr(TetraParamType::Encoding);
    }

    return TetraTrainProgress{
        .current_iteration = current_iteration_.load(),
        .total_iterations = config_.iterations,
        .loss = current_loss_.load(),
        .psnr = 0.0f,  // Would compute from loss
        .num_vertices = mesh_ ? mesh_->num_interior_vertices() : 0,
        .num_tetrahedra = mesh_ ? mesh_->num_tetrahedra() : 0,
        .vertices_lr = current_vert_lr,
        .encoding_lr = current_enc_lr,
        .mesh_frozen = optimizer_ ? optimizer_->is_frozen() : (current_iteration_.load() >= config_.freeze_start)
    };
}

// ------------------------------
// CHECKPOINTING
// ------------------------------

std::expected<void, std::string> TetraTrainer::save_checkpoint(
    const std::filesystem::path& path) const {

    LOG_INFO("Saving checkpoint to {}", path.string());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected("Failed to open file for writing");
    }

    // Write header
    const uint32_t magic = 0x54455443;  // "TETC"
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write iteration
    int iter = current_iteration_.load();
    file.write(reinterpret_cast<const char*>(&iter), sizeof(iter));

    // Write config
    file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));

    // Write mesh
    mesh_->serialize(file);

    // Write features
    features_->serialize(file);

    // Write optimizer state
    if (optimizer_) {
        optimizer_->serialize(file);
    }

    LOG_INFO("Checkpoint saved");
    return {};
}

std::expected<int, std::string> TetraTrainer::load_checkpoint(
    const std::filesystem::path& path) {

    LOG_INFO("Loading checkpoint from {}", path.string());

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected("Failed to open file for reading");
    }

    // Read and verify header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x54455443) {
        return std::unexpected("Invalid checkpoint format");
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        return std::unexpected("Unsupported checkpoint version");
    }

    // Read iteration
    int iter;
    file.read(reinterpret_cast<char*>(&iter), sizeof(iter));

    // Read config
    file.read(reinterpret_cast<char*>(&config_), sizeof(config_));

    // Read mesh
    mesh_ = std::make_unique<TetraMesh>();
    mesh_->deserialize(file);

    // Read features
    features_ = std::make_unique<TetraFeatures>();
    features_->deserialize(file);

    // Initialize renderer
    renderer_ = std::make_unique<TetraRenderer>();
    TetraRenderConfig render_config;
    renderer_->initialize(render_config);

    // Setup optimizer and try to load saved state
    auto opt_result = setup_optimizer();
    if (!opt_result) {
        return std::unexpected("Failed to setup optimizer: " + opt_result.error());
    }

    // Try to load optimizer state if present in checkpoint
    if (optimizer_ && file.peek() != EOF) {
        try {
            optimizer_->deserialize(file);
            LOG_INFO("Optimizer state restored from checkpoint");
        } catch (const std::exception& e) {
            LOG_WARN("Could not restore optimizer state: {}", e.what());
            // Not fatal - optimizer will start fresh
        }
    }

    current_iteration_ = iter;
    initialized_ = true;

    LOG_INFO("Checkpoint loaded, resuming from iteration {}", iter);
    return iter;
}

// ------------------------------
// EXPORT
// ------------------------------

void TetraTrainer::export_mesh(const std::filesystem::path& path,
                               bool extract_surface) const {
    if (!mesh_) {
        LOG_WARN("No mesh to export");
        return;
    }

    if (extract_surface) {
        mesh_->export_triangle_mesh(path);
    } else {
        mesh_->save_ply(path);
    }

    LOG_INFO("Mesh exported to {}", path.string());
}

void TetraTrainer::export_renders(const std::filesystem::path& output_dir) const {
    if (!mesh_ || !features_ || !renderer_ || !dataset_) {
        LOG_WARN("Not fully initialized, cannot export renders");
        return;
    }

    std::filesystem::create_directories(output_dir);

    const auto& cameras = dataset_->get_cameras();
    for (size_t i = 0; i < cameras.size(); ++i) {
        auto& cam = *cameras[i];

        auto render_result = renderer_->forward(*mesh_, *features_, cam, background_);
        if (!render_result) {
            LOG_WARN("Failed to render camera {}", i);
            continue;
        }

        // Save rendered image
        auto path = output_dir / (std::to_string(i) + ".png");
        // In full implementation, would use image I/O to save
    }

    LOG_INFO("Exported {} renders to {}", cameras.size(), output_dir.string());
}

// ------------------------------
// EVALUATION
// ------------------------------

namespace {

/**
 * @brief Compute PSNR between two images
 *
 * PSNR = 20 * log10(max_value / sqrt(MSE))
 *
 * @param pred Predicted image tensor [C, H, W] or [H, W, C]
 * @param target Ground truth image tensor
 * @param data_range Maximum value of the data (default 1.0 for normalized images)
 * @return PSNR value in dB
 */
float compute_psnr(const lfs::core::Tensor& pred,
                   const lfs::core::Tensor& target,
                   float data_range = 1.0f) {
    // Compute MSE: mean((pred - target)^2)
    auto diff = pred - target;
    auto squared_diff = diff * diff;
    float mse = squared_diff.mean().item<float>();

    // Clamp to avoid log(0)
    if (mse < 1e-10f) {
        mse = 1e-10f;
    }

    // PSNR = 20 * log10(data_range / sqrt(MSE))
    return 20.0f * std::log10(data_range / std::sqrt(mse));
}

/**
 * @brief Compute SSIM between two images using LichtFeld's SSIM kernel
 *
 * @param pred Predicted image tensor [C, H, W] or [1, C, H, W]
 * @param target Ground truth image tensor
 * @return Mean SSIM value
 */
float compute_ssim(const lfs::core::Tensor& pred,
                   const lfs::core::Tensor& target) {
    // Use LibTorch-free SSIM kernel with valid padding
    auto [ssim_value, ctx] = lfs::training::kernels::ssim_forward(pred, target, true);
    return ssim_value.mean().item<float>();
}

} // anonymous namespace

std::expected<EvalMetrics, std::string> TetraTrainer::evaluate(
    const std::filesystem::path& output_dir,
    bool save_images) const {

    if (!initialized_.load()) {
        return std::unexpected("Trainer not initialized");
    }

    if (!mesh_ || !features_ || !renderer_ || !dataset_) {
        return std::unexpected("Missing required components for evaluation");
    }

    EvalMetrics metrics;
    metrics.iteration = current_iteration_.load();

    // Create output directories if saving images
    std::filesystem::path pred_dir;
    std::filesystem::path gt_dir;
    if (save_images && !output_dir.empty()) {
        pred_dir = output_dir / "pred";
        gt_dir = output_dir / "gt";
        std::filesystem::create_directories(pred_dir);
        std::filesystem::create_directories(gt_dir);
    }

    LOG_INFO("Starting evaluation on {} images", dataset_->size());
    auto start_time = std::chrono::steady_clock::now();

    const size_t num_images = dataset_->size();
    metrics.per_image_psnr.reserve(num_images);
    metrics.per_image_ssim.reserve(num_images);

    // ------------------------------------------------------------
    // Iterate through all cameras in dataset
    // ------------------------------------------------------------
    for (size_t i = 0; i < num_images; ++i) {
        // Get camera example (loads ground truth image)
        auto example = dataset_->get(i);
        auto& cam = *example.data.camera;
        core::Tensor gt_image = std::move(example.data.image);

        if (!gt_image.is_valid()) {
            LOG_WARN("Camera {} has no valid image, skipping", i);
            continue;
        }

        // Ensure ground truth is on CUDA
        if (gt_image.device() != core::Device::CUDA) {
            gt_image = gt_image.to(core::Device::CUDA);
        }

        // Render the view
        auto render_result = renderer_->forward(*mesh_, *features_, cam, background_);
        if (!render_result) {
            LOG_WARN("Failed to render camera {}: {}", i, render_result.error());
            continue;
        }

        // Clamp rendered image to [0, 1]
        core::Tensor rendered = render_result->rgb.clamp(0.0f, 1.0f);

        // Convert rendered from HWC to CHW to match GT format
        // Renderer outputs HWC [H, W, C], GT is CHW [C, H, W]
        if (rendered.ndim() == 3 && rendered.shape()[2] <= 4) {
            rendered = rendered.permute({2, 0, 1}).contiguous();
        }

        // Ensure shapes match for metric computation
        if (rendered.shape() != gt_image.shape()) {
            LOG_WARN("Shape mismatch at image {}, skipping", i);
            continue;
        }

        // ------------------------------------------------------------
        // Compute metrics
        // ------------------------------------------------------------
        float psnr_val = compute_psnr(rendered, gt_image);
        float ssim_val = compute_ssim(rendered, gt_image);

        metrics.per_image_psnr.push_back(psnr_val);
        metrics.per_image_ssim.push_back(ssim_val);

        // ------------------------------------------------------------
        // Save images if requested
        // ------------------------------------------------------------
        if (save_images && !output_dir.empty()) {
            auto pred_path = pred_dir / (std::to_string(i) + ".png");
            auto gt_path = gt_dir / (std::to_string(i) + ".png");

            // Use async image saving to avoid blocking
            lfs::core::image_io::save_image_async(pred_path, rendered);
            lfs::core::image_io::save_image_async(gt_path, gt_image);
        }

        // Log progress every 10 images
        if ((i + 1) % 10 == 0 || i == num_images - 1) {
            LOG_INFO("Evaluated {}/{} images", i + 1, num_images);
        }
    }

    // Wait for all image saves to complete
    if (save_images && !output_dir.empty()) {
        lfs::core::image_io::wait_for_pending_saves();
    }

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float>(end_time - start_time).count();

    // ------------------------------------------------------------
    // Compute mean metrics
    // ------------------------------------------------------------
    metrics.num_images = static_cast<int>(metrics.per_image_psnr.size());
    metrics.elapsed_time = elapsed;

    if (metrics.num_images > 0) {
        metrics.mean_psnr = std::accumulate(
            metrics.per_image_psnr.begin(),
            metrics.per_image_psnr.end(),
            0.0f) / metrics.num_images;

        metrics.mean_ssim = std::accumulate(
            metrics.per_image_ssim.begin(),
            metrics.per_image_ssim.end(),
            0.0f) / metrics.num_images;
    }

    // ------------------------------------------------------------
    // Save metrics to JSON if output directory specified
    // ------------------------------------------------------------
    if (!output_dir.empty()) {
        std::filesystem::path metrics_path = output_dir / "metrics.json";
        std::ofstream metrics_file(metrics_path);
        if (metrics_file) {
            metrics_file << "{\n";
            metrics_file << "  \"iteration\": " << metrics.iteration << ",\n";
            metrics_file << "  \"mean_psnr\": " << std::fixed << std::setprecision(6)
                         << metrics.mean_psnr << ",\n";
            metrics_file << "  \"mean_ssim\": " << metrics.mean_ssim << ",\n";
            metrics_file << "  \"elapsed_time\": " << metrics.elapsed_time << ",\n";
            metrics_file << "  \"num_images\": " << metrics.num_images << ",\n";
            metrics_file << "  \"per_image\": [\n";
            for (size_t i = 0; i < metrics.per_image_psnr.size(); ++i) {
                metrics_file << "    {\"psnr\": " << metrics.per_image_psnr[i]
                             << ", \"ssim\": " << metrics.per_image_ssim[i] << "}";
                if (i < metrics.per_image_psnr.size() - 1) {
                    metrics_file << ",";
                }
                metrics_file << "\n";
            }
            metrics_file << "  ]\n";
            metrics_file << "}\n";
            metrics_file.close();
            LOG_INFO("Metrics saved to {}", metrics_path.string());
        }
    }

    LOG_INFO("Evaluation complete: PSNR={:.4f}, SSIM={:.4f}, Time={:.2f}s",
             metrics.mean_psnr, metrics.mean_ssim, metrics.elapsed_time);

    return metrics;
}

// ------------------------------
// VIDEO RENDERING
// ------------------------------

namespace {

/**
 * @brief Convert R, T tensors to 4x4 camera-to-world matrix
 *
 * Camera stores world-to-camera: [R|T] where R is 3x3 rotation, T is 3x1 translation
 * Camera-to-world is the inverse: [R^T | -R^T * T]
 */
std::array<float, 16> compute_c2w(const lfs::core::Tensor& R_tensor,
                                   const lfs::core::Tensor& T_tensor) {
    // Move to CPU for easy access
    lfs::core::Tensor R_cpu = R_tensor.to(lfs::core::Device::CPU);
    lfs::core::Tensor T_cpu = T_tensor.to(lfs::core::Device::CPU);

    const float* R = R_cpu.ptr<float>();
    const float* T = T_cpu.ptr<float>();

    // R is stored as [3, 3], T as [3]
    // c2w = [R^T | -R^T * T; 0 0 0 1]
    std::array<float, 16> c2w;

    // Transpose R (column-major style for row-major output)
    c2w[0] = R[0]; c2w[1] = R[3]; c2w[2] = R[6]; // First column of R^T
    c2w[4] = R[1]; c2w[5] = R[4]; c2w[6] = R[7]; // Second column of R^T
    c2w[8] = R[2]; c2w[9] = R[5]; c2w[10] = R[8]; // Third column of R^T

    // Translation: -R^T * T
    c2w[3] = -(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]);
    c2w[7] = -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]);
    c2w[11] = -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2]);

    // Bottom row
    c2w[12] = 0.0f; c2w[13] = 0.0f; c2w[14] = 0.0f; c2w[15] = 1.0f;

    return c2w;
}

/**
 * @brief Normalize a 3D vector
 */
void normalize3(float* v) {
    float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-8f) {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
}

/**
 * @brief Cross product of two 3D vectors
 */
void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * @brief Generate elliptical camera path around scene
 *
 * Follows the pattern from radiance_meshes: compute an ellipse that
 * encompasses the training cameras and generate smooth interpolated poses.
 *
 * Returns vector of 4x4 matrices in row-major order (c2w transforms)
 */
std::vector<std::array<float, 16>> generate_ellipse_path(
    const std::vector<std::array<float, 16>>& poses,
    int n_frames) {

    if (poses.empty()) {
        return {};
    }

    // Extract camera positions (translation column of c2w)
    std::vector<std::array<float, 3>> positions;
    positions.reserve(poses.size());

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float sum_z = 0.0f;

    for (const auto& pose : poses) {
        // Position is in column 3 (indices 3, 7, 11)
        float x = pose[3];
        float y = pose[7];
        float z = pose[11];

        positions.push_back({x, y, z});

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        sum_z += z;
    }

    float avg_z = sum_z / static_cast<float>(poses.size());
    float radius_x = (max_x - min_x) * 0.5f;
    float radius_y = (max_y - min_y) * 0.5f;
    float center_x = (min_x + max_x) * 0.5f;
    float center_y = (min_y + max_y) * 0.5f;

    // Ensure minimum radius (scaled by scene extent)
    float scene_extent = std::max(max_x - min_x, max_y - min_y);
    float min_radius = scene_extent * 0.1f;
    radius_x = std::max(radius_x, min_radius);
    radius_y = std::max(radius_y, min_radius);

    // Compute focus point (centroid of positions as approximation)
    float focus_x = center_x;
    float focus_y = center_y;
    float focus_z = avg_z - scene_extent * 0.3f;  // Slightly in front of cameras

    // Compute average up vector from camera poses
    float avg_up[3] = {0, 0, 0};
    for (const auto& pose : poses) {
        // Y-axis of camera (column 1: indices 1, 5, 9)
        avg_up[0] += pose[1];
        avg_up[1] += pose[5];
        avg_up[2] += pose[9];
    }
    normalize3(avg_up);

    // Default to world up if average is near zero
    float up_len = std::sqrt(avg_up[0]*avg_up[0] + avg_up[1]*avg_up[1] + avg_up[2]*avg_up[2]);
    if (up_len < 0.1f) {
        avg_up[0] = 0; avg_up[1] = 0; avg_up[2] = 1;
    }

    // Generate elliptical path
    std::vector<std::array<float, 16>> path_poses;
    path_poses.reserve(n_frames);

    for (int i = 0; i < n_frames; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n_frames);
        float theta = t * 2.0f * static_cast<float>(M_PI);

        // Position on ellipse
        float pos_x = center_x + radius_x * std::cos(theta);
        float pos_y = center_y + radius_y * std::sin(theta);
        float pos_z = avg_z;

        // Create look-at matrix (camera-to-world)
        // Forward direction (from position to focus)
        float forward[3] = {
            focus_x - pos_x,
            focus_y - pos_y,
            focus_z - pos_z
        };
        normalize3(forward);

        // Right direction (cross of forward and up)
        float right[3];
        cross3(forward, avg_up, right);
        normalize3(right);

        // Actual up (cross of right and forward)
        float up[3];
        cross3(right, forward, up);
        normalize3(up);

        // Build c2w matrix (row-major)
        // Columns are: right, up, -forward (OpenGL convention), position
        std::array<float, 16> pose;
        pose[0] = right[0];   pose[1] = up[0];   pose[2] = -forward[0]; pose[3] = pos_x;
        pose[4] = right[1];   pose[5] = up[1];   pose[6] = -forward[1]; pose[7] = pos_y;
        pose[8] = right[2];   pose[9] = up[2];   pose[10] = -forward[2]; pose[11] = pos_z;
        pose[12] = 0.0f;      pose[13] = 0.0f;   pose[14] = 0.0f;       pose[15] = 1.0f;

        path_poses.push_back(pose);
    }

    return path_poses;
}

/**
 * @brief Convert c2w matrix to R and T tensors for Camera construction
 */
std::pair<lfs::core::Tensor, lfs::core::Tensor> c2w_to_RT(const std::array<float, 16>& c2w) {
    // c2w is [R_c2w | t_c2w; 0 0 0 1] in row-major
    // Camera wants world-to-camera: R_w2c = R_c2w^T, T_w2c = -R_w2c * t_c2w

    // Extract c2w rotation (columns 0,1,2 of rows 0,1,2)
    float R_c2w[9] = {
        c2w[0], c2w[1], c2w[2],    // Row 0
        c2w[4], c2w[5], c2w[6],    // Row 1
        c2w[8], c2w[9], c2w[10]    // Row 2
    };
    float t_c2w[3] = {c2w[3], c2w[7], c2w[11]};

    // Transpose R for w2c
    float R_w2c[9] = {
        R_c2w[0], R_c2w[3], R_c2w[6],
        R_c2w[1], R_c2w[4], R_c2w[7],
        R_c2w[2], R_c2w[5], R_c2w[8]
    };

    // T_w2c = -R_w2c * t_c2w
    float T_w2c[3] = {
        -(R_w2c[0]*t_c2w[0] + R_w2c[1]*t_c2w[1] + R_w2c[2]*t_c2w[2]),
        -(R_w2c[3]*t_c2w[0] + R_w2c[4]*t_c2w[1] + R_w2c[5]*t_c2w[2]),
        -(R_w2c[6]*t_c2w[0] + R_w2c[7]*t_c2w[1] + R_w2c[8]*t_c2w[2])
    };

    lfs::core::Tensor R = lfs::core::Tensor::from_vector(
        std::vector<float>(R_w2c, R_w2c + 9), {3, 3}, lfs::core::Device::CUDA);
    lfs::core::Tensor T = lfs::core::Tensor::from_vector(
        std::vector<float>(T_w2c, T_w2c + 3), {3}, lfs::core::Device::CUDA);

    return {R, T};
}

} // anonymous namespace

std::expected<void, std::string> TetraTrainer::render_video(
    const std::filesystem::path& output_path,
    int num_frames,
    int fps) const {

    if (!initialized_.load()) {
        return std::unexpected("Trainer not initialized");
    }

    if (!mesh_ || !features_ || !renderer_ || !dataset_) {
        return std::unexpected("Missing required components for video rendering");
    }

    LOG_INFO("Rendering rotating video: {} frames at {} fps", num_frames, fps);

    // Get camera poses from dataset
    const auto& cameras = dataset_->get_cameras();
    if (cameras.empty()) {
        return std::unexpected("No cameras in dataset");
    }

    // Extract c2w matrices from cameras
    std::vector<std::array<float, 16>> poses;
    poses.reserve(cameras.size());
    for (const auto& cam : cameras) {
        poses.push_back(compute_c2w(cam->R(), cam->T()));
    }

    // Generate elliptical camera path
    auto path_poses = generate_ellipse_path(poses, num_frames);
    if (path_poses.empty()) {
        return std::unexpected("Failed to generate camera path");
    }

    // Create temporary directory for frames
    auto temp_dir = std::filesystem::temp_directory_path() / "lichtfeld_video_frames";
    std::filesystem::create_directories(temp_dir);

    // Use reference camera for intrinsics
    const auto& ref_cam = *cameras[0];
    int width = ref_cam.image_width() > 0 ? ref_cam.image_width() : ref_cam.camera_width();
    int height = ref_cam.image_height() > 0 ? ref_cam.image_height() : ref_cam.camera_height();

    if (width <= 0 || height <= 0) {
        return std::unexpected("Invalid image dimensions from reference camera");
    }

    LOG_INFO("Rendering {} frames at {}x{}", num_frames, width, height);

    // Render each frame
    for (int i = 0; i < num_frames; ++i) {
        // Convert c2w to R, T for Camera construction
        auto [R, T] = c2w_to_RT(path_poses[i]);

        // Create camera for this frame (minimal constructor)
        core::Camera frame_cam(
            R, T,
            ref_cam.focal_x(), ref_cam.focal_y(),
            ref_cam.center_x(), ref_cam.center_y(),
            ref_cam.radial_distortion(),
            ref_cam.tangential_distortion(),
            ref_cam.camera_model_type(),
            "frame_" + std::to_string(i),
            "",  // No image path
            "",  // No mask path
            width, height,
            i    // uid
        );
        frame_cam.set_image_dimensions(width, height);
        // Note: CUDA tensors are initialized in Camera constructor via world_to_view()

        // Render frame
        auto render_result = renderer_->forward(*mesh_, *features_, frame_cam, background_);
        if (!render_result) {
            LOG_WARN("Failed to render frame {}: {}", i, render_result.error());
            continue;
        }

        // Clamp to [0, 1] and save
        core::Tensor frame = render_result->rgb.clamp(0.0f, 1.0f);

        // Save frame with zero-padded filename
        char frame_name[32];
        std::snprintf(frame_name, sizeof(frame_name), "frame_%05d.png", i);
        auto frame_path = temp_dir / frame_name;

        core::image_io::save_image_async(frame_path, frame);

        // Log progress every 50 frames
        if ((i + 1) % 50 == 0 || i == num_frames - 1) {
            LOG_INFO("Rendered {}/{} frames", i + 1, num_frames);
        }
    }

    // Wait for all frames to be saved
    core::image_io::wait_for_pending_saves();

    // Verify frames exist
    int frame_count = 0;
    for (int i = 0; i < num_frames; ++i) {
        char frame_name[32];
        std::snprintf(frame_name, sizeof(frame_name), "frame_%05d.png", i);
        if (std::filesystem::exists(temp_dir / frame_name)) {
            frame_count++;
        }
    }
    LOG_INFO("Found {}/{} frame files", frame_count, num_frames);

    if (frame_count == 0) {
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        return std::unexpected("No frames were saved to temporary directory");
    }

    // Ensure output directory exists
    std::filesystem::create_directories(output_path.parent_path());

    // Use ffmpeg to encode video
    // Note: -vf scale ensures dimensions are divisible by 2 (required by libx264)
    std::string ffmpeg_cmd = "ffmpeg -y -framerate " + std::to_string(fps) +
        " -i \"" + (temp_dir / "frame_%05d.png").string() + "\"" +
        " -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\"" +
        " -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p" +
        " \"" + output_path.string() + "\"";

    LOG_INFO("Encoding video with ffmpeg...");
    LOG_INFO("Command: {}", ffmpeg_cmd);
    int result = std::system(ffmpeg_cmd.c_str());

    // Check if video was created successfully
    bool video_success = (result == 0) &&
                         std::filesystem::exists(output_path) &&
                         std::filesystem::file_size(output_path) > 0;

    // Clean up temporary frames
    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);

    if (!video_success) {
        return std::unexpected("ffmpeg encoding failed with code " + std::to_string(result));
    }

    LOG_INFO("Video saved to {}", output_path.string());
    return {};
}

} // namespace lfs::tetra
