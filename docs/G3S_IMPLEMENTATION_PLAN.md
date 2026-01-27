# Geometry-Grounded Gaussian Splatting (G³S) Implementation Plan

## Overview

This plan implements the G³S paper in LichtFeld-Studio, enabling better geometry extraction through stochastic solid depth rendering.

**Paper:** [Geometry-Grounded Gaussian Splatting](https://arxiv.org/abs/2601.17835)

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         G³S IMPLEMENTATION OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: CUDA KERNELS (Parallel)                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ A: Vacancy      │  │ B: Stochastic   │  │ C: Binary       │             │
│  │    Field        │  │    Transmit.    │  │    Search       │             │
│  │    v(x)=√(1-G)  │  │    T_i(t)       │  │    T=0.5        │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                ▼                                            │
│  PHASE 2: GRADIENT KERNELS ────────────────────────────────────────────────│
│  ┌─────────────────────────────────────────┐                               │
│  │ D: Closed-Form Depth Gradient           │                               │
│  │    ∂t_med/∂θ = -[∂T/∂θ] / [∂T/∂t]       │                               │
│  └─────────────────────────────────────────┘                               │
│                                │                                            │
│                                ▼                                            │
│  PHASE 3: TRAINING INTEGRATION (Parallel)                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ E: Depth Loss   │  │ F: Multi-View   │  │ G: Normal       │             │
│  │    Kernel       │  │    Regularizer  │  │    Consistency  │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                ▼                                            │
│  PHASE 4: PIPELINE INTEGRATION                                             │
│  ┌─────────────────────────────────────────┐                               │
│  │ H: Trainer Integration                   │                               │
│  │    - RenderMode::G3S_DEPTH              │                               │
│  │    - Loss integration                    │                               │
│  │    - Configuration                       │                               │
│  └─────────────────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: CUDA Forward Kernels (Parallel Workstreams)

### Task A: Vacancy Field Computation
**File:** `src/training/kernels/g3s_vacancy.cu` (NEW)

**Math:**
```
v(x) = √(1 - G(x))
where G(x) = o·exp(-(x-x_c)ᵀΣ⁻¹(x-x_c))
```

**Interface:**
```cpp
namespace lfs::training::kernels {
    // Compute vacancy at arbitrary 3D points
    void launch_compute_vacancy(
        const float* means,      // [N, 3] Gaussian centers
        const float* quats,      // [N, 4] Quaternions
        const float* scales,     // [N, 3] Log-scales
        const float* opacities,  // [N] Log-opacities
        const float* query_pts,  // [M, 3] Query points
        float* vacancy_out,      // [M] Output vacancy values
        int N, int M,
        cudaStream_t stream);
}
```

**Dependencies:** None (can start immediately)
**Estimated complexity:** Low

---

### Task B: Stochastic Transmittance Computation
**File:** `src/training/kernels/g3s_transmittance.cu` (NEW)

**Math (Equation 12 from paper):**
```
T_i(t) = {
    v_i(t),                    if t ≤ t*_i
    v_i(t*_i)² / v_i(t),       if t > t*_i
}
where t*_i is the Gaussian's peak along the ray
```

**Interface:**
```cpp
namespace lfs::training::kernels {
    // Per-Gaussian transmittance at depth t
    __device__ __forceinline__ float compute_stochastic_transmittance(
        float t,           // Query depth
        float t_star,      // Peak depth (from RaDe-GS)
        float vacancy_at_t_star,
        float vacancy_at_t);

    // Accumulated transmittance along ray
    void launch_compute_ray_transmittance(
        const float* depths,         // [C, M] Per-gaussian depths
        const float* t_stars,        // [C, M] Peak depths
        const float* vacancies,      // [C, M] Vacancy at peak
        const int32_t* tile_offsets, // Tile ranges
        const int32_t* flatten_ids,  // Sorted gaussian indices
        float* transmittance_out,    // [C, H, W] Per-pixel transmittance
        float query_t,               // Depth to evaluate at
        int C, int H, int W,
        cudaStream_t stream);
}
```

**Dependencies:** Task A (vacancy computation)
**Estimated complexity:** Medium

---

### Task C: Binary Search for Median Depth
**File:** `src/training/kernels/g3s_median_depth.cu` (NEW)

**Algorithm:**
```
1. Initialize interval [t_init - r, t_init + r] where r=0.4
2. For 5 iterations:
   a. Divide into 8 segments (7 sample points)
   b. Compute T at each sample point
   c. Find segment where T crosses 0.5
   d. Narrow interval to that segment
3. Return midpoint as t_med
```

**Interface:**
```cpp
namespace lfs::training::kernels {
    void launch_g3s_median_depth_forward(
        // Input
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* init_depths,    // [C, H, W] RaDe-GS initial estimate
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        // Output
        float* median_depths,        // [C, H, W] G³S median depth
        bool* valid_mask,            // [C, H, W] Pixels where T crosses 0.5
        // Config
        float search_radius,         // Default: 0.4
        int num_iterations,          // Default: 5
        int C, int H, int W, int N,
        cudaStream_t stream);
}
```

**Dependencies:** Task A, Task B
**Estimated complexity:** High

---

## Phase 2: CUDA Backward Kernels

### Task D: Closed-Form Depth Gradient
**File:** `src/training/kernels/g3s_median_depth_backward.cu` (NEW)

**Math (Equation 13 from paper):**
```
∂t_med/∂θ = -[∂T(t_med; θ)/∂θ] / [∂T(t; θ)/∂t]|_{t=t_med}
```

**Key insight:** Gradient distributes to ALL Gaussians along ray, not just one.

**Interface:**
```cpp
namespace lfs::training::kernels {
    void launch_g3s_median_depth_backward(
        // Forward inputs (cached)
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* median_depths,
        const bool* valid_mask,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        // Upstream gradient
        const float* grad_depth,     // [C, H, W] dL/d(t_med)
        // Output gradients
        float* grad_means,           // [N, 3]
        float* grad_quats,           // [N, 4]
        float* grad_scales,          // [N, 3]
        float* grad_opacities,       // [N]
        // Config
        int C, int H, int W, int N,
        cudaStream_t stream);
}
```

**Dependencies:** Tasks A, B, C (needs forward pass structure)
**Estimated complexity:** High

---

## Phase 3: Training Losses (Parallel Workstreams)

### Task E: Depth Supervision Loss
**File:** `src/training/losses/depth_loss.hpp` (NEW)

**Interface:**
```cpp
namespace lfs::training {
    struct DepthLossConfig {
        float weight = 1.0f;
        bool use_l1 = true;
        bool use_gradient_smoothness = false;
    };

    class DepthLoss {
    public:
        Tensor forward(
            const Tensor& rendered_depth,  // [H, W]
            const Tensor& gt_depth,        // [H, W] (optional)
            const Tensor& valid_mask);     // [H, W]

        void backward(
            const Tensor& grad_loss,
            Tensor& grad_depth);
    };
}
```

**Dependencies:** None (can start in parallel)
**Estimated complexity:** Low

---

### Task F: Multi-View Geometric Regularization
**File:** `src/training/losses/multiview_regularization.hpp` (NEW)

**Math (from PGSR):**
```
L_gc = Σ w(u_r) · ψ(u_r)
where ψ(u_r) = ||u_r - H_nr·H_rn·u_r||₂  (cycle reprojection error)
      w(u_r) = exp(-ψ(u_r)) if ψ < 1, else 0
```

**Interface:**
```cpp
namespace lfs::training {
    struct MultiViewRegConfig {
        float photometric_weight = 0.6f;
        float geometric_weight = 0.02f;
        int patch_size = 11;  // NCC window
    };

    class MultiViewRegularization {
    public:
        Tensor compute_loss(
            const Camera& ref_cam,
            const Camera& neighbor_cam,
            const Tensor& ref_depth,
            const Tensor& ref_normal,
            const Tensor& ref_image,
            const Tensor& neighbor_image);
    };
}
```

**Dependencies:** Depth rendering (Task C)
**Estimated complexity:** Medium-High

---

### Task G: Normal Consistency Loss
**File:** `src/training/losses/normal_consistency.hpp` (NEW)

**Math:**
```
L_n = Σ ω_i · (1 - n_i · ñ)
where n_i = Gaussian normal (shortest axis)
      ñ = normal from depth gradient
      ω_i = alpha blending weight
```

**Interface:**
```cpp
namespace lfs::training {
    class NormalConsistencyLoss {
    public:
        Tensor forward(
            const Tensor& gaussian_normals,  // [N, 3]
            const Tensor& depth_normals,     // [H, W, 3]
            const Tensor& blend_weights);    // [H, W, N]
    };
}
```

**Dependencies:** Depth rendering (Task C)
**Estimated complexity:** Medium

---

## Phase 4: Pipeline Integration

### Task H: Trainer Integration
**Files to modify:**
- `src/core/include/core/parameters.hpp` - Add G³S config
- `src/training/trainer.cpp` - Add loss integration
- `src/training/rasterization/gsplat_rasterizer.cpp` - Add G3S mode

**Configuration:**
```cpp
struct G3SParameters {
    bool enabled = false;
    float depth_loss_weight = 1.0f;
    float multiview_weight = 0.6f;
    float normal_consistency_weight = 0.05f;
    float search_radius = 0.4f;
    int binary_search_iterations = 5;
    int start_iteration = 7000;  // Enable geometric reg after warmup
};
```

**RenderMode Extension:**
```cpp
enum class GsplatRenderMode {
    RGB,
    ED,
    RGB_D,
    RGB_ED,
    G3S_DEPTH,     // NEW: G³S stochastic depth
    RGB_G3S_DEPTH  // NEW: RGB + G³S depth
};
```

**Dependencies:** All previous tasks
**Estimated complexity:** Medium

---

## Parallel Execution Strategy

```
TIME ──────────────────────────────────────────────────────────────────►

Agent 1 (CUDA Forward):
├── Task A: Vacancy Field ────────┤
│                                 │
│   Task B: Stochastic Transmit. ─┼──────────┤
│                                            │
│   Task C: Binary Search ───────────────────┼──────────────┤

Agent 2 (CUDA Backward):
│   [wait for A,B,C]                                        │
│                                                           │
├── Task D: Depth Gradient ─────────────────────────────────┼─────────┤

Agent 3 (Losses - No Dependencies):
├── Task E: Depth Loss ───────────┤
│                                 │
├── Task F: Multi-View Reg ───────┼──────────────────────────────────┤
│                                 │
├── Task G: Normal Consistency ───┼─────────────────┤

Agent 4 (Integration):
│   [wait for all above]                                              │
│                                                                     │
├── Task H: Trainer Integration ──────────────────────────────────────┼──┤

                                                                      DONE
```

---

## File Structure

```
src/training/
├── kernels/
│   ├── g3s_vacancy.cu              # Task A (NEW)
│   ├── g3s_vacancy.cuh             # Task A (NEW)
│   ├── g3s_transmittance.cu        # Task B (NEW)
│   ├── g3s_transmittance.cuh       # Task B (NEW)
│   ├── g3s_median_depth.cu         # Task C (NEW)
│   ├── g3s_median_depth.cuh        # Task C (NEW)
│   └── g3s_median_depth_backward.cu # Task D (NEW)
├── losses/
│   ├── depth_loss.hpp              # Task E (NEW)
│   ├── depth_loss.cpp              # Task E (NEW)
│   ├── multiview_regularization.hpp # Task F (NEW)
│   ├── multiview_regularization.cpp # Task F (NEW)
│   ├── normal_consistency.hpp      # Task G (NEW)
│   └── normal_consistency.cpp      # Task G (NEW)
└── trainer.cpp                     # Task H (MODIFY)

src/core/include/core/
└── parameters.hpp                  # Task H (MODIFY: add G3SParameters)

src/rendering/rasterizer/gsplat_fwd/
└── RasterizeToPixelsFromWorld3DGSFwd.cu  # Reference only
```

---

## Key Code Locations (Reference)

| Component | File | Lines |
|-----------|------|-------|
| Current median depth | `src/rendering/rasterizer/gsplat_fwd/RasterizeToPixelsFromWorld3DGSFwd.cu` | 284-286 |
| Transmittance update | Same file | 280 |
| Alpha computation | Same file | 272-275 |
| Trainer loss loop | `src/training/trainer.cpp` | 1146-1206 |
| Regularization kernels | `src/training/kernels/regularization.cu` | All |
| Photometric loss | `src/training/losses/photometric_loss.cpp` | All |
| SplatData | `src/core/include/core/splat_data.hpp` | All |
| Parameters | `src/core/include/core/parameters.hpp` | OptimizationParameters |

---

## Testing Strategy

1. **Unit tests per kernel:**
   - Vacancy: Compare with analytical solution
   - Transmittance: Verify against discrete baseline
   - Binary search: Check convergence to T=0.5

2. **Integration tests:**
   - Forward pass: Render depth, compare with baseline median
   - Backward pass: Finite difference gradient check

3. **End-to-end:**
   - Train on DTU dataset subset
   - Compare Chamfer distance with baseline

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| DTU Chamfer Distance | ~0.74 (GOF) | ≤0.61 (G³S paper) |
| Training time (DTU) | ~52 min | ≤35 min |
| Multi-view consistency | Poor | Strong (cycle error < 1px) |
| Depth map quality | Jagged edges | Smooth, sharp boundaries |
