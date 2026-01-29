# gDel3D integration for LichtFeld-Studio (no PyTorch dependency)
# GPU Delaunay triangulation library
#
# This file is included from external/CMakeLists.txt
# Path to gDel3D submodule is passed via GDEL3D_SUBMODULE_DIR

if(NOT DEFINED GDEL3D_SUBMODULE_DIR)
    message(FATAL_ERROR "GDEL3D_SUBMODULE_DIR must be set before including gDel3D.cmake")
endif()

set(GDEL3D_SOURCES
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/RandGen.cpp
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/CPU/Splaying.cpp
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/CPU/Star.cpp
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/CPU/predicates.cpp
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/CPU/PredWrapper.cpp
)

set(GDEL3D_CUDA_SOURCES
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/GpuDelaunay.cu
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/GPU/KerDivision.cu
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/GPU/KerPredicates.cu
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D/GPU/ThrustWrapper.cu
)

add_library(gDel3D STATIC
    ${GDEL3D_SOURCES}
    ${GDEL3D_CUDA_SOURCES}
)

target_include_directories(gDel3D
    PUBLIC
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d
    ${GDEL3D_SUBMODULE_DIR}/src/gdel3d/gDel3D
)

target_link_libraries(gDel3D
    PUBLIC
    CUDA::cudart
)

# Use double precision for Delaunay (required for numerical stability)
# NOTE: pyGDel3D (used by radiance_meshes) also uses float64
# The orient3d determinant calculation requires high precision to avoid
# false "coplanar" detection for nearly-coplanar SfM points
# target_compile_definitions(gDel3D PRIVATE REAL_TYPE_FP32)  # Don't use - causes degeneracy errors

set_target_properties(gDel3D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

message(STATUS "gDel3D (GPU Delaunay) configured for LichtFeld-Studio")
