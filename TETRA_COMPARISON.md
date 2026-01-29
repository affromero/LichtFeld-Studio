# Radiance Meshes vs LichtFeld TETRA Comparison Log

Generated: 2026-01-29
Dataset: /home/ubuntu/Code/Hax-CV/data/Nerfstudio/kitchen

## Critical Finding: PSNR Gap Root Cause

**Python PSNR @ 200 iters: 16.60**
**C++ PSNR @ 30 iters: 8.27**

### Root Cause: Point Count Cap

| Metric | Python (radiance_meshes) | C++ (LichtFeld) |
|--------|--------------------------|-----------------|
| Initial Points | 241,367 (COLMAP) | 241,367 (COLMAP) |
| **Used Points** | **142,073** | **10,000** ← CRITICAL |
| Tetrahedra | 881,572 | 24,592 |
| Point Reduction | 41% | **95.8%** |

**Location of cap in C++:**
```cpp
// src/tetra/tetra_mesh.cpp:71
constexpr size_t MAX_POINTS = 10000;  // TOO LOW!
```

---

## Python Configuration (radiance_meshes/train.py)

### Hash Grid / iNGP Settings
```python
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-4
args.hidden_dim = 64
args.scale_multi = 0.35
args.log2_hashmap_size = 23      # 8,388,608 entries
args.per_level_scale = 2
args.L = 8                        # 8 levels (not 16!)
args.hashmap_dim = 8              # Features per level (not 2!)
args.base_resolution = 64
args.density_offset = -4
```

### Vertex Optimization Settings
```python
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.delaunay_interval = 10
```

### Densification Settings
```python
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000           # Budget for densification
args.clone_min_contrib = 0.003
args.split_min_contrib = 0.01
```

### Loss Settings
```python
args.lambda_ssim = 0.2           # SSIM weight
# L1 weight = 1 - 0.2 = 0.8 (implicit)
```

### Freeze Schedule
```python
args.freeze_start = 18000        # Freeze mesh after this
args.freeze_lr = 1e-3
args.final_freeze_lr = 1e-4
```

---

## C++ Configuration (LichtFeld TETRA)

### tetra_trainer.hpp defaults
```cpp
// Hash grid
HashGridConfig hash_config;      // default L=16, hashmap_size=2^21

// MLP config
MLPConfig mlp_config;            // default hidden_dim=64

// Learning rates
float vertices_lr = 1e-4f;
float encoding_lr = 3e-3f;
float network_lr = 1e-3f;

// Loss weights
float lambda_l1 = 0.8f;
float lambda_ssim = 0.2f;

// Densification
int densify_start = 2000;
int densify_end = 16000;
int densify_interval = 500;
float densify_grad_threshold = 0.0002f;

// Mesh update
int delaunay_interval = 10;
int freeze_start = 18000;
```

---

## Key Differences Found

### 1. **Number of Hash Grid Levels**
- Python: `L = 8` levels
- C++: `L = 16` levels (tetra_features.hpp:20)

### 2. **Hash Grid Feature Dimension**
- Python: `hashmap_dim = 8` features per level (train.py:96)
- C++: `features_per_level = 2` (tetra_features.hpp:21)

### 3. **Hash Map Size**
- Python: `log2_hashmap_size = 23` (8,388,608 entries per level)
- C++: `log2_hashmap_size = 19` (524,288 entries per level)

### 4. **Base Resolution**
- Python: `base_resolution = 64` (train.py:97)
- C++: `base_resolution = 16` (tetra_features.hpp:23)

### 3. **Point Count (CRITICAL)**
- Python: Uses all available points (142K for kitchen)
- C++: Caps at 10,000 points

### 4. **Scene Scaling**
- Python: `scene_scale = 4.426` (computed)
- C++: `scene_scale = 69.77` (computed differently)

---

## Training Output Comparison

### Python @ 200 iterations
```
Scene scaling: 4.426436901092529
PSNR=16.60
#V=142073
#T=881572
DL=1.56
```

### C++ @ 30 iterations
```
Scene scale: 69.77
loss=0.38 (approx)
10008 vertices
24592 tetrahedra
PSNR (eval): 8.27
SSIM (eval): 0.27
```

---

## Recommendations to Fix C++ Implementation

### Priority 1: Increase Point Cap (CRITICAL)
```cpp
// Change in tetra_mesh.cpp:71
// OLD: constexpr size_t MAX_POINTS = 10000;
constexpr size_t MAX_POINTS = 300000;  // or remove cap entirely
```
This alone should improve PSNR significantly.

### Priority 2: Match Hash Grid Config
```cpp
// Change in tetra_features.hpp:19-26
struct HashGridConfig {
    int num_levels = 8;               // Match Python's L=8 (was 16)
    int features_per_level = 8;       // Match Python's hashmap_dim=8 (was 2)
    int log2_hashmap_size = 23;       // Match Python (was 19)
    int base_resolution = 64;         // Match Python (was 16)
    float per_level_scale = 2.0f;     // Same
    int max_resolution = 8192;        // Same
};

// Update MLPConfig accordingly:
struct MLPConfig {
    int input_features = 64;          // L * F = 8 * 8 (was 32)
    int hidden_dim = 64;              // Same
    // ...
};
```

### Priority 3: Fix Scene Scaling
The Python uses a different scene normalization that results in scale ~4.4 vs 69.77 in C++.

Check `scene/transforms.py` in radiance_meshes to see how they compute scene scale.

---

## Exact Parameter Comparison Table

| Parameter | Python (radiance_meshes) | C++ (LichtFeld) | Match? |
|-----------|-------------------------|-----------------|--------|
| **Points** | 142,073 | 10,000 | ❌ |
| **Tetrahedra** | 881,572 | 24,592 | ❌ |
| num_levels (L) | 8 | 16 | ❌ |
| features_per_level (F) | 8 | 2 | ❌ |
| log2_hashmap_size | 23 | 19 | ❌ |
| base_resolution | 64 | 16 | ❌ |
| per_level_scale | 2.0 | 2.0 | ✅ |
| hidden_dim | 64 | 64 | ✅ |
| vertices_lr | 1e-4 | 1e-4 | ✅ |
| encoding_lr | 3e-3 | 3e-3 | ✅ |
| network_lr | 1e-3 | 1e-3 | ✅ |
| lambda_ssim | 0.2 | 0.2 | ✅ |
| densify_start | 2000 | 2000 | ✅ |
| densify_end | 16000 | 16000 | ✅ |
| densify_interval | 500 | 500 | ✅ |
| delaunay_interval | 10 | 10 | ✅ |
| freeze_start | 18000 | 18000 | ✅ |

---

## Full Training Log (Python)

See: full_training.log

Key excerpts:
- CameraInfo example shows fovx=0.898, fovy=0.620
- Image resolution after downscale: 3115x2078 / 2 = 1557x1039
- Training split: 279 images (no explicit test split in this run)

---

## Files for Reproducibility

1. `/tmp/radiance_meshes_reference/full_training.log` - Complete console output
2. `/tmp/radiance_meshes_reference/metrics.json` - Final metrics
3. `/tmp/radiance_meshes_reference/rotating.mp4` - Rotating video (400 frames)
4. `/tmp/radiance_meshes_reference/ckpt.ply` - Final mesh
5. `/tmp/radiance_meshes_reference/ckpt.pth` - PyTorch checkpoint
