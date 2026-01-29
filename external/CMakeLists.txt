# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies (EXCLUDE_FROM_ALL to prevent third-party install rules)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/CMakeLists.txt")
    message(STATUS "Adding ImGuizmo")
    add_subdirectory(imguizmo EXCLUDE_FROM_ALL)
else()
    message(WARNING "ImGuizmo not found - gizmo controls unavailable")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec")
    if(DEFINED DETECTED_COMPUTE_CAP)
        string(REPLACE "." "" _arch "${DETECTED_COMPUTE_CAP}")
        set(CUDA_TARGET_ARCHS "${_arch}" CACHE STRING "" FORCE)
    endif()
    set(BUILD_NVJPEG_EXT ON CACHE BOOL "" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "" FORCE)
    add_subdirectory(nvImageCodec EXCLUDE_FROM_ALL)
else()
    message(WARNING "nvImageCodec not found - GPU image loading unavailable")
endif()

# gDel3D - GPU Delaunay triangulation for tetra module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gDel3D/src/gdel3d/gDel3D/GpuDelaunay.cu")
    message(STATUS "Adding gDel3D (GPU Delaunay)")
    set(GDEL3D_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gDel3D")
    include("${CMAKE_SOURCE_DIR}/cmake/external/gDel3D.cmake")
else()
    message(WARNING "gDel3D not found - GPU Delaunay unavailable, using CPU fallback")
    message(WARNING "  Run: git submodule update --init external/gDel3D")
endif()
