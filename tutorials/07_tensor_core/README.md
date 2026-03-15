# Lesson 07: Tensor Core

本教程介绍 Tensor Core 编程和 WMMA API。

## 学习目标

- 理解 Tensor Core 的工作原理
- 掌握 WMMA API 的使用
- 学习 warp 级别的协作编程
- 比较 Tensor Core 和 CUDA Core 的性能差异

## 硬件要求

**需要 Volta 架构或更新的 GPU：**

| 架构 | GPU 示例 | 计算能力 | Tensor Core 支持 |
|------|----------|----------|------------------|
| Volta | V100 | 7.0 | ✓ FP16 |
| Turing | RTX 2080 | 7.5 | ✓ FP16/INT8 |
| Ampere | RTX 3060/3080/3090 | 8.0/8.6 | ✓ FP16/BF16/TF32 |
| Ada | RTX 4080/4090 | 8.9/9.0 | ✓ FP8/FP16/BF16/TF32 |

> 本教程针对 **RTX 3060 Laptop GPU (SM 8.6)** 配置，使用 half 精度。

## 文件结构

```
07_tensor_core/
├── tensor_core.cu          # TODO 练习代码
├── solution/
│   └── tensor_core.cu      # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

```bash
cd /mnt/e/small-project
mkdir -p build && cd build
cmake ..
make -j
```

可执行文件位于：
- `tutorials/07_tensor_core/tensor_core` - TODO 版本
- `tutorials/07_tensor_core/tensor_core_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/07_tensor_core

# 编译 TODO 版本（需要指定架构）
nvcc -o tensor_core tensor_core.cu -I../../include -arch=sm_86

# 编译参考答案
nvcc -o tensor_core_solution solution/tensor_core.cu -I../../include -arch=sm_86
```

## 运行方法

```bash
# 运行 TODO 版本
./tensor_core

# 运行参考答案
./tensor_core_solution
```

### 预期输出

```
=== Lesson 07: Tensor Core ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

Tensor Core 支持：是 (SM 8.6)

矩阵大小：1024 x 1024
启动配置：16x16 threads, 64x64 grid

测试 1: CUDA Core GEMM
----------------------
时间：38.170 ms

测试 2: Tensor Core WMMA
----------------------
时间：0.255 ms

加速比：约 125x
```

## 练习任务

1. **声明 WMMA Fragment**：定义 matrix_a、matrix_b、accumulator
2. **加载数据**：从全局内存加载到 fragment
3. **执行 MMA**：调用 `wmma::mma_sync`
4. **存储结果**：将结果写回全局内存

## 关键概念

### 什么是 Tensor Core？

Tensor Core 是 NVIDIA GPU 中的专用矩阵乘法单元，专门用于深度学习中的矩阵运算。

```
Tensor Core 执行：D = A × B + C

其中 A、B、C、D 都是 4×4×4 矩阵（Volta/Turing）
或 16×16×16 矩阵（Ampere+，使用 WMMA API）
```

### Tensor Core 性能优势

| 精度 | 每时钟周期操作数 | 相比 FP32 |
|------|------------------|-----------|
| FP32 (CUDA Core) | 1 | 1x |
| FP16 (Tensor Core) | 64 | 64x |
| TF32 (Ampere+) | 32 | 32x |
| BF16 (Ampere+) | 64 | 64x |
| INT8 (Turing+) | 128 | 128x |
| FP8 (Hopper+) | 256 | 256x |

### WMMA API 基础

```cpp
using namespace nvcuda;

// 1. 声明 WMMA fragments
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 2. 初始化 accumulator
wmma::fill_fragment(c_frag, 0.0f);

// 3. 加载数据（从全局内存到 fragment）
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);

// 4. 执行矩阵乘法
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 5. 存储结果
wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
```

### Warp 协作

WMMA API 需要整个 warp（32 个线程）协作：

```
Warp 0: 负责 C[0:16, 0:16] 块
  - Thread 0-31: 协作加载、计算、存储

Warp 1: 负责 C[0:16, 16:32] 块
...
```

## 练习答案提示

<details>
<summary>WMMA 矩阵乘法实现要点</summary>

```cpp
__global__ void tensor_core_wmma(half* A, half* B, float* C, int n) {
    // 计算 warp ID
    int warpId = (threadIdx.x / 32) + (blockIdx.x * blockDim.y + threadIdx.y) * (blockDim.x / 32);

    // 计算输出块的行列索引
    int block_row = warpId / ((n + 16) / 16);
    int block_col = warpId % ((n + 16) / 16);

    int row_start = block_row * 16;
    int col_start = block_col * 16;

    // 声明 fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化
    wmma::fill_fragment(c_frag, 0.0f);

    // 边界检查后执行 mma_sync
    if (row_start < n && col_start < n) {
        // 加载数据、执行计算、存储结果
        // ...（详细代码见 solution/tensor_core.cu）
    }
}
```

</details>

## 常见问题

**Q: 为什么我的 GPU 不支持 Tensor Core？**
A: Tensor Core 需要 Volta (7.0) 或更新的架构。查看 [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)。

**Q: half 精度会有精度损失吗？**
A: 是的，FP16 的数值范围比 FP32 小，可能导致溢出或下溢。解决方案是用 FP32 存储结果。

**Q: 为什么 WMMA 需要 warp 协作？**
A: Tensor Core 是 warp 级别的指令，需要 32 个线程协作才能发挥性能。

**Q: TF32 和 FP32 有什么区别？**
A: TF32 有 FP32 的动态范围和 FP16 的精度，是 Ampere 架构引入的新格式。

## 性能提示

1. **使用合适的精度**：在精度允许的情况下优先使用 FP16/BF16
2. **对齐内存访问**：确保矩阵大小是 16 的倍数
3. **使用共享内存**：减少全局内存访问延迟
4. **批量处理**：使用 cuBLAS 的批量 GEMM 获得更好性能

## 下一步

完成本教程后，继续学习：
- [Lesson 08: cuBLAS 库](../08_cublas/README.md)
