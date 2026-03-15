cmake_minimum_required(VERSION 3.18)
project(CUDA_Tutorial LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 检测 CUDA 架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 自动检测当前 GPU 的架构
    include(CheckLanguage)
    check_language(CUDA)

    # 默认支持常见架构
    set(CMAKE_CUDA_ARCHITECTURES
        60 61 62  # Pascal
        70 72 75  # Volta, Turing
        80 86     # Ampere
        89        # Ada Lovelace
    )
endif()

# 设置 CUDA 编译器
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# 查找 CUDA 包
find_package(CUDAToolkit QUIET)

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# 查找 cuBLAS（可选）
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
find_path(CUBLAS_INCLUDE_DIR cublas_v2.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)

if(CUBLAS_LIBRARIES AND CUBLAS_INCLUDE_DIR)
    set(CUBLAS_FOUND TRUE)
    message(STATUS "cuBLAS found: ${CUBLAS_LIBRARIES}")
else()
    set(CUBLAS_FOUND FALSE)
    message(STATUS "cuBLAS not found - Lesson 08 will be skipped")
endif()

# 包含目录（使用绝对路径）
set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
include_directories(${PROJECT_INCLUDE_DIR})

# 输出配置信息
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")

# 添加子目录
add_subdirectory(tutorials/00_basics)
add_subdirectory(tutorials/01_vector_add)
add_subdirectory(tutorials/02_matrix_mul)
add_subdirectory(tutorials/03_shared_memory)
add_subdirectory(tutorials/04_atomic_ops)
add_subdirectory(tutorials/05_streams)
add_subdirectory(tutorials/06_cuda_graph)
add_subdirectory(tutorials/07_tensor_core)
if(CUBLAS_FOUND)
    add_subdirectory(tutorials/08_cublas)
else()
    message(STATUS "Skipping tutorials/08_cublas (cuBLAS not found)")
endif()
add_subdirectory(tutorials/09_optimization)

# 自定义目标：编译所有 TODO 代码
add_custom_target(build_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMENT "Building all tutorial targets"
)
