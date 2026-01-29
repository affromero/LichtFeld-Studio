# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# tetra module - Tetrahedral mesh representation (Radiance Meshes)
# ============================================================================

# CUDA kernel sources
set(TETRA_CUDA_SOURCES
    cuda/tetra_intersect.cu
    cuda/tetra_rasterize.cu
    cuda/tetra_rays.cu
    cuda/tetra_tile_sort.cu
    cuda/hash_grid.cu
    cuda/mlp_kernels.cu
)

# C++ sources
set(TETRA_SOURCES
    tetra_mesh.cpp
    tetra_features.cpp
    tetra_renderer.cpp
    tetra_delaunay.cpp
    tetra_trainer.cpp
    tetra_optimizer.cpp
)

# Create static library
add_library(lfs_tetra STATIC
    ${TETRA_SOURCES}
    ${TETRA_CUDA_SOURCES}
)

target_include_directories(lfs_tetra
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(lfs_tetra
    PUBLIC
        lfs_core           # Tensor, Camera, PointCloud
        lfs_io             # COLMAP loader
        lfs_training       # Adam optimizer, losses
        lfs_geometry       # Geometry utilities
        fastlfs_backend    # CUDA primitives
        CUDA::cudart
        CUDA::cublas
    PRIVATE
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        glm::glm
)

# Optional: GPU Delaunay via gDel3D
if(TARGET gDel3D)
    target_link_libraries(lfs_tetra PRIVATE gDel3D)
    target_compile_definitions(lfs_tetra PRIVATE LFS_HAS_GPU_DELAUNAY=1)
    set(LFS_TETRA_GPU_DELAUNAY ON)
else()
    set(LFS_TETRA_GPU_DELAUNAY OFF)
endif()

# CUDA properties
# Note: CUDA_RESOLVE_DEVICE_SYMBOLS OFF to avoid duplicate symbol errors
# when gDel3D is linked - device symbols resolved at final executable link
set_target_properties(lfs_tetra PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler options
if(MSVC)
    target_compile_options(lfs_tetra PRIVATE
        $<$<CONFIG:Debug>:/Od /Z7>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(lfs_tetra PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )
endif()

# CUDA compiler flags
target_compile_options(lfs_tetra PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

message(STATUS "")
message(STATUS "===============================================================")
message(STATUS "  tetra - Tetrahedral Mesh Module (Radiance Meshes)")
message(STATUS "===============================================================")
message(STATUS "  Namespace: lfs::tetra")
message(STATUS "  Features: Delaunay triangulation, hash grid encoding, tile-based rasterization")
if(LFS_TETRA_GPU_DELAUNAY)
    message(STATUS "  GPU Delaunay: ENABLED (gDel3D)")
else()
    message(STATUS "  GPU Delaunay: DISABLED (CPU fallback)")
endif()
message(STATUS "")
