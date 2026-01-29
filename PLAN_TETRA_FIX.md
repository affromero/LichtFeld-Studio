# Plan: Fix C++ TETRA Renderer to Match Python Quality

## Problem Summary

The C++ TETRA renderer produces garbage output (orange triangles) because the `compute_intersection_colors_kernel` uses placeholder depth-based coloring instead of actual feature queries.

**Current State:**
- PSNR: ~8.5 (garbage)
- Speed: ~5.4 it/s
- Output: Orange triangles (depth-based placeholder)

**Target State:**
- PSNR: ~16.6 (match Python)
- Speed: >10 it/s (faster than Python)
- Output: Correct rendered images

---

## Root Cause Analysis

### Python Implementation (radiance_meshes)
Uses **per-tet linear color model** with 7 parameters per tetrahedron:
- `density`: scalar (opacity control)
- `base_color`: float3 (DC color)
- `gradient`: float3 (spatial color variation)

**Color formula:**
```
color(x) = base_color + dot(gradient, position)
```

**Volume integration:** Spline-based analytic integration with exponential transmittance.

### C++ Implementation (LichtFeld-Studio)
Currently has two different architectures:
1. `TetraFeatures`: Hash grid + MLP (instant-NGP style) - **NOT what Python uses**
2. `TetraPlyFeatures`: Storage format matching Python - **correct format but not used in rendering**

**The Bug:** The renderer calls `compute_intersection_colors_kernel` which uses depth-based coloring instead of querying features.

---

## Solution Architecture

### Option A: Implement Per-Tet Linear Model (RECOMMENDED)
Match Python exactly by implementing per-tet parameters.

**Pros:**
- Exact match with Python quality
- Simpler model (fewer parameters)
- Proven to work (Python PSNR ~16.6)

**Cons:**
- Need to add per-tet parameter storage
- Need to implement linear color model in CUDA

### Option B: Wire Existing TetraFeatures
Use the hash grid + MLP approach already in C++.

**Pros:**
- Code already exists
- More expressive model

**Cons:**
- Different architecture than Python
- May not match Python quality
- More complex

**Decision: Option A** - to match Python exactly as user requested.

---

## Implementation Plan

### Phase 1: Add Per-Tet Parameters to TetraMesh

**File: `src/tetra/include/tetra/tetra_mesh.hpp`**

Add to TetraMesh class:
```cpp
// Per-tetrahedron learnable parameters
core::Tensor density_;     // [T] float32 - opacity control
core::Tensor base_color_;  // [T, 3] float32 - DC color
core::Tensor gradient_;    // [T, 3] float32 - spatial color variation
```

Add methods:
```cpp
// Parameter access
core::Tensor& density() { return density_; }
core::Tensor& base_color() { return base_color_; }
core::Tensor& gradient() { return gradient_; }

// Initialize parameters from SH (for loading Python models)
void init_parameters_from_sh(const TetraPlyFeatures& features);

// Initialize random parameters (for training from scratch)
void init_parameters_random();
```

### Phase 2: Update CUDA Color Computation Kernel

**File: `src/tetra/cuda/tetra_intersect.cu`**

Replace `compute_intersection_colors_kernel` with:

```cpp
__global__ void compute_intersection_colors_kernel(
    // Geometry
    const float* __restrict__ vertices,      // [V, 3]
    const int64_t* __restrict__ tetrahedra,  // [T, 4]
    const int64_t* __restrict__ tet_indices, // [H*W, max_K]
    const float* __restrict__ barycentrics,  // [H*W, max_K, 4]
    const float* __restrict__ depths,        // [H*W, max_K]
    const int32_t* __restrict__ num_intersects, // [H*W]

    // Per-tet parameters (NEW)
    const float* __restrict__ tet_density,   // [T]
    const float* __restrict__ tet_base_color,// [T, 3]
    const float* __restrict__ tet_gradient,  // [T, 3]

    // Output
    float* __restrict__ out_rgb,             // [H*W, max_K, 3]
    float* __restrict__ out_alpha,           // [H*W, max_K]

    // Params
    int width, int height, int max_K)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pixel_idx = py * width + px;
    int num_k = num_intersects[pixel_idx];

    for (int k = 0; k < max_K; ++k) {
        int idx = pixel_idx * max_K + k;

        if (k < num_k) {
            int64_t tet_idx = tet_indices[idx];

            // Get per-tet parameters
            float density = tet_density[tet_idx];
            float3 base_color = make_float3(
                tet_base_color[tet_idx * 3 + 0],
                tet_base_color[tet_idx * 3 + 1],
                tet_base_color[tet_idx * 3 + 2]);
            float3 gradient = make_float3(
                tet_gradient[tet_idx * 3 + 0],
                tet_gradient[tet_idx * 3 + 1],
                tet_gradient[tet_idx * 3 + 2]);

            // Get intersection position from barycentrics
            float4 bary = make_float4(
                barycentrics[idx * 4 + 0],
                barycentrics[idx * 4 + 1],
                barycentrics[idx * 4 + 2],
                barycentrics[idx * 4 + 3]);

            int4 tet = make_int4(
                tetrahedra[tet_idx * 4 + 0],
                tetrahedra[tet_idx * 4 + 1],
                tetrahedra[tet_idx * 4 + 2],
                tetrahedra[tet_idx * 4 + 3]);

            float3 v0 = make_float3(vertices[tet.x*3], vertices[tet.x*3+1], vertices[tet.x*3+2]);
            float3 v1 = make_float3(vertices[tet.y*3], vertices[tet.y*3+1], vertices[tet.y*3+2]);
            float3 v2 = make_float3(vertices[tet.z*3], vertices[tet.z*3+1], vertices[tet.z*3+2]);
            float3 v3 = make_float3(vertices[tet.w*3], vertices[tet.w*3+1], vertices[tet.w*3+2]);

            // Position = barycentric interpolation
            float3 pos = bary.x * v0 + bary.y * v1 + bary.z * v2 + bary.w * v3;

            // Linear color model: color = base_color + dot(gradient, pos)
            float color_offset = dot(gradient, pos);
            float3 color = base_color + make_float3(color_offset, color_offset, color_offset);
            color = fmaxf(color, make_float3(0.0f, 0.0f, 0.0f)); // clamp to [0, inf)

            // Alpha from density * segment length (approximation)
            float segment_length = (k + 1 < num_k) ?
                (depths[pixel_idx * max_K + k + 1] - depths[idx]) : 0.1f;
            float alpha = 1.0f - expf(-density * segment_length);

            out_rgb[idx * 3 + 0] = color.x;
            out_rgb[idx * 3 + 1] = color.y;
            out_rgb[idx * 3 + 2] = color.z;
            out_alpha[idx] = alpha;
        } else {
            out_rgb[idx * 3 + 0] = 0.0f;
            out_rgb[idx * 3 + 1] = 0.0f;
            out_rgb[idx * 3 + 2] = 0.0f;
            out_alpha[idx] = 0.0f;
        }
    }
}
```

### Phase 3: Update Renderer to Pass Per-Tet Parameters

**File: `src/tetra/tetra_renderer.cpp`**

Update `forward()` to pass per-tet parameters:

```cpp
// Step 7: Compute RGB and alpha using per-tet linear color model
cuda::launch_compute_intersection_colors(
    all_verts_gpu.ptr<float>(),
    tets_gpu.ptr<int64_t>(),
    intersection_ids.ptr<int64_t>(),
    barycentric_coords.ptr<float>(),
    ray_depths.ptr<float>(),
    num_intersects.ptr<int32_t>(),
    mesh.density().ptr<float>(),      // NEW
    mesh.base_color().ptr<float>(),   // NEW
    mesh.gradient().ptr<float>(),     // NEW
    sample_rgb.ptr<float>(),
    sample_alpha.ptr<float>(),
    width, height, max_K,
    nullptr);
```

### Phase 4: Update Function Signatures

**File: `src/tetra/include/tetra/tetra_renderer.hpp`**

Update CUDA declaration:
```cpp
void launch_compute_intersection_colors(
    const float* vertices,
    const int64_t* tetrahedra,
    const int64_t* tet_indices,
    const float* barycentrics,
    const float* depths,
    const int32_t* num_intersects,
    const float* tet_density,     // NEW
    const float* tet_base_color,  // NEW
    const float* tet_gradient,    // NEW
    float* out_rgb,
    float* out_alpha,
    int width, int height, int max_K,
    void* stream);
```

### Phase 5: Update Trainer to Optimize Per-Tet Parameters

**File: `src/tetra/tetra_trainer.cpp`**

Add per-tet parameters to optimizer:
```cpp
std::expected<void, std::string> TetraTrainer::setup_optimizer() {
    // Existing vertex parameters...

    // Add per-tet parameters
    optimizer_->add_param_group(
        mesh_->density(), config_.encoding_lr, "density");
    optimizer_->add_param_group(
        mesh_->base_color(), config_.encoding_lr, "base_color");
    optimizer_->add_param_group(
        mesh_->gradient(), config_.encoding_lr, "gradient");
}
```

### Phase 6: Implement Backward Pass for Per-Tet Parameters

**File: `src/tetra/cuda/tetra_intersect.cu`**

Add backward kernel:
```cpp
__global__ void compute_intersection_colors_backward_kernel(
    // Forward inputs
    const float* vertices, const int64_t* tetrahedra,
    const int64_t* tet_indices, const float* barycentrics,
    const float* depths, const int32_t* num_intersects,
    const float* tet_density, const float* tet_base_color,
    const float* tet_gradient,

    // Backward inputs (gradients from alpha_blend_backward)
    const float* grad_rgb, const float* grad_alpha,

    // Backward outputs
    float* grad_density,     // [T]
    float* grad_base_color,  // [T, 3]
    float* grad_gradient,    // [T, 3]
    float* grad_vertices,    // [V, 3] (optional, for vertex optimization)

    int width, int height, int max_K);
```

---

## Testing Plan

### Test 1: Visual Sanity Check
After Phase 2, run a single forward pass and check that:
- Output is no longer orange triangles
- Colors vary based on position

### Test 2: Gradient Check
After Phase 6, verify gradients with finite differences:
```cpp
// Perturb density[0] by epsilon
// Check (loss(+eps) - loss(-eps)) / (2*eps) ≈ grad_density[0]
```

### Test 3: Training Convergence
Run 200 iterations and verify:
- Loss decreases monotonically
- PSNR improves
- Rendered images look reasonable

### Test 4: Match Python Quality
Run full training and compare:
- Final PSNR should be ~16.6 (same as Python)
- Training should be >10 it/s (faster than Python)

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `tetra_mesh.hpp` | Modify | Add per-tet parameter tensors |
| `tetra_mesh.cpp` | Modify | Initialize per-tet parameters |
| `tetra_intersect.cu` | Modify | Update color kernel to use per-tet params |
| `tetra_renderer.hpp` | Modify | Update CUDA function signatures |
| `tetra_renderer.cpp` | Modify | Pass per-tet params to CUDA kernel |
| `tetra_trainer.cpp` | Modify | Add per-tet params to optimizer |

---

## Estimated Effort

- Phase 1: 30 min (add parameter storage)
- Phase 2: 45 min (update CUDA kernel)
- Phase 3: 15 min (update renderer)
- Phase 4: 10 min (update signatures)
- Phase 5: 20 min (update trainer)
- Phase 6: 45 min (backward pass)

**Total: ~2.5 hours**

---

## Risks and Mitigations

1. **Risk:** Alpha computation differs from Python
   - **Mitigation:** Study Python's spline integration carefully

2. **Risk:** Backward pass numerical stability
   - **Mitigation:** Use atomic operations and gradient clipping

3. **Risk:** Memory layout differences
   - **Mitigation:** Ensure tensors are contiguous before CUDA calls
