# Lesson 07: Tensor Core

## 学习目标

- 理解 Tensor Core 架构和原理
- 学习使用 WMMA API 编程
- 掌握 Tensor Core 矩阵乘法
- 了解精度和性能权衡

## Tensor Core 简介

### 什么是 Tensor Core？

Tensor Core 是 NVIDIA Volta 架构引入的**专用矩阵乘法单元**。

```
┌─────────────────────────────┐
│    Streaming Multiprocessor │
│  ┌───────┐ ┌───────┐        │
│  │  FP32 │ │  FP32 │  CUDA  │
│  │ Core  │ │ Core  │  Core │
│  └───────┘ └───────┘        │
│  ┌─────────────────────┐    │
│  │   Tensor Core       │    │ ← 专用矩阵单元
│  │   (4x4x4)           │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
```

### 架构演进

| 架构 | Tensor Core | 精度支持 | 吞吐 (FP16) |
|------|-------------|----------|-------------|
| Volta V100 | 64/SM | FP16, INT8 | 125 TOPS |
| Turing RTX20 | 64/SM | FP16, INT8, INT4 | 230 TOPS |
| Ampere RTX30 | 64/SM | FP16, BF16, TF32 | 238 TOPS |
| Ampere A100 | 64/SM | FP64, FP16, BF16, TF32, INT8 | 312 TOPS |
| Hopper H100 | 64/SM | FP8, FP16, BF16, TF32 | 989 TOPS |

## WMMA API 基础

### 什么是 WMMA？

WMMA (Warp Matrix Multiply Accumulate) 是 NVIDIA 提供的 Tensor Core 编程 API。

### 基本类型

```cpp
#include <mma.h>
using namespace nvcuda;

// 矩阵片段
wmma::fragment<wmma::matrix_a, M, N, K, precision, layout> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, precision, layout> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
```

### 支持的精度

| 精度 | 类型 | 用途 |
|------|------|------|
| `wmma::precision::fp16` | half | AI 训练/推理 |
| `wmma::precision::tf32` | tf32 | AI 训练 |
| `wmma::precision::bf16` | bfloat16 | AI 训练 |
| `wmma::precision::fp32` | float | 高精度计算 |

## WMMA 矩阵乘法

### 基本流程

```cpp
__global__ void wmma_gemm(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 声明 fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 2. 初始化 accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // 3. 加载数据
    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B, ldb);

    // 4. 执行矩阵乘法
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 5. 存储结果
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}
```

### 尺寸要求

WMMA 要求矩阵是 **16 的倍数**：

- M, N, K 必须能被 16 整除
- 最小尺寸：16×16×16

## 共享内存优化

### 使用共享内存加载数据

```cpp
__global__ void wmma_shared_gemm(float* A, float* B, float* C, int M, int N, int K) {
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    const int BLOCK_K = 16;

    // 共享内存
    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    // Fragment
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int tid = threadIdx.x;

    // 加载到共享内存
    for (int k = tid; k < BLOCK_K; k += blockDim.x) {
        for (int m = 0; m < BLOCK_M; m++) {
            shared_a[m][k] = A[(block_m * BLOCK_M + m) * K + k];
        }
        for (int n = 0; n < BLOCK_N; n++) {
            shared_b[k][n] = B[k * N + block_n * BLOCK_N + n];
        }
    }
    __syncthreads();

    // 加载到 fragment
    wmma::load_matrix_sync(a_frag, shared_a, BLOCK_K);
    wmma::load_matrix_sync(b_frag, shared_b, BLOCK_N);

    // 执行 MMA
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    wmma::store_matrix_sync(C + block_m * BLOCK_M * N + block_n * BLOCK_N,
                           c_frag, N, wmma::mem_row_major);
}
```

## 性能对比

### Tensor Core vs CUDA Core

| 操作 | CUDA Core | Tensor Core | 加速 |
|------|-----------|-------------|------|
| FP16 GEMM | 8 TFLOPS | 125 TFLOPS | 15x |
| TF32 GEMM | 8 TFLOPS | 156 TFLOPS | 19x |
| FP32 GEMM | 8 TFLOPS | 78 TFLOPS (TF32) | 9x |

### 实测数据 (RTX 4090)

```
矩阵大小：1024 x 1024 x 1024

CUDA Core (FP32):     2.5 ms
Tensor Core (TF32):   0.3 ms  (8.3x 加速)
Tensor Core (FP16):   0.15 ms (16.7x 加速)
```

## 精度对比

### TF32 vs FP32

| 特性 | FP32 | TF32 |
|------|------|------|
| 有效位数 | 23 | 10 |
| 指数位数 | 8 | 8 |
| 动态范围 | 高 | 高 |
| 精度 | 7 位小数 | 3 位小数 |
| 性能 | 1x | 8x |

### 选择建议

- **AI 训练**: TF32 或 BF16
- **AI 推理**: FP16 或 INT8
- **科学计算**: FP32 或 FP64 (A100)

## 实战：完整的 GEMM

### 多 Block 实现

```cpp
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tensor_core_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ float sa[BLOCK_M * BLOCK_K];
    __shared__ float sb[BLOCK_K * BLOCK_N];

    int tid = threadIdx.x;
    int out_row = block_m * BLOCK_M;
    int out_col = block_n * BLOCK_N;

    // 边界检查
    if (out_row >= M || out_col >= N) return;

    // 分块 K
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // 加载 A
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int r = i / BLOCK_K;
            int c = i % BLOCK_K;
            sa[i] = (out_row + r < M && k_tile + c < K) ?
                    A[(out_row + r) * K + k_tile + c] : 0.0f;
        }

        // 加载 B
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            sb[i] = (k_tile + r < K && out_col + c < N) ?
                    B[(k_tile + r) * N + out_col + c] : 0.0f;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, sa, BLOCK_K);
        wmma::load_matrix_sync(b_frag, sb, BLOCK_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(C + out_row * N + out_col, c_frag, N, wmma::mem_row_major);
}
```

## 练习

### TODO: 完成 tensor_core.cu

1. 检查 GPU 是否支持 Tensor Core
2. 实现基本的 WMMA 矩阵乘法
3. 比较与 CUDA Core 的性能差异

### 进阶练习

1. 实现共享内存优化版本
2. 测试不同精度的性能和精度
3. 实现批量的矩阵乘法

## 调试工具

### 检查 Tensor Core 使用

```bash
ncu --metrics tensor__active ./your_program
```

### 性能分析

```bash
nsys profile --stats=true ./your_program
```

## 常见问题

### Q1: 我的 GPU 支持 Tensor Core 吗？

检查计算能力：
- 需要 7.0+ (Volta 或更新)

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
```

### Q2: WMMA 最小尺寸是多少？

16×16×16，矩阵必须是 16 的倍数。

### Q3: 如何选择精度？

- 速度优先：FP16
- 平衡：TF32
- 精度优先：FP32

## 下一步

完成 [tensor_core.cu](../tutorials/07_tensor_core/tensor_core.cu) 的 TODO 部分，然后学习 [Lesson 08: cuBLAS](08_cublas.md)。
