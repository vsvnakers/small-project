# Lesson 01: 向量加法

本教程介绍 CUDA 内存管理和基本的并行计算模式。

## 学习目标

- 掌握 CUDA 内存管理（分配、复制、释放）
- 理解 Host 和 Device 内存的区别
- 学习一维线程索引计算
- 了解边界检查的重要性

## 文件结构

```
01_vector_add/
├── vector_add.cu           # TODO 练习代码（你需要补全）
├── solution/
│   └── vector_add.cu       # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

在项目根目录构建：

```bash
cd /mnt/e/small-project
mkdir -p build && cd build
cmake ..
make -j
```

编译完成后，可执行文件位于：
- `tutorials/01_vector_add/vector_add` - TODO 版本
- `tutorials/01_vector_add/vector_add_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/01_vector_add

# 编译 TODO 版本
nvcc -o vector_add vector_add.cu -I../../include

# 编译参考答案
nvcc -o vector_add_solution solution/vector_add.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本（补全代码后）
./vector_add

# 运行参考答案
./vector_add_solution
```

### 预期输出

```
=== Lesson 01: 向量加法 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

向量大小：1000000
启动 kernel: 3907 blocks x 256 threads

结果验证（前 10 个元素）:
  h_c[0] = 0.0 (期望：0.0)
  h_c[1] = 3.0 (期望：3.0)
  h_c[2] = 6.0 (期望：6.0)
  h_c[3] = 9.0 (期望：9.0)
  h_c[4] = 12.0 (期望：12.0)
  h_c[5] = 15.0 (期望：15.0)
  h_c[6] = 18.0 (期望：18.0)
  h_c[7] = 21.0 (期望：21.0)
  h_c[8] = 24.0 (期望：24.0)
  h_c[9] = 27.0 (期望：27.0)

程序完成!
```

## 练习任务

1. **分配 Device 内存**：使用 `cudaMalloc` 为三个向量分配 GPU 内存
2. **内存拷贝**：使用 `cudaMemcpy` 将数据从 CPU 传到 GPU
3. **实现 Kernel**：补全 `vector_add_kernel` 函数
4. **返回结果**：将计算结果从 GPU 传回 CPU
5. **清理资源**：使用 `cudaFree` 释放 GPU 内存

## 关键概念

### CUDA 内存管理 API

```cpp
// 1. 分配 device 内存
cudaMalloc(&d_ptr, size_in_bytes);

// 2. Host -> Device 拷贝
cudaMemcpy(d_dest, h_src, size, cudaMemcpyHostToDevice);

// 3. Device -> Host 拷贝
cudaMemcpy(h_dest, d_src, size, cudaMemcpyDeviceToHost);

// 4. 释放 device 内存
cudaFree(d_ptr);
```

### Kernel 索引计算

```cpp
// 一维线程索引
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 边界检查

```cpp
// 始终检查索引，防止越界访问
if (idx < n) {
    // 安全的内存访问
}
```

### Grid-Stride 计算

```cpp
int threads_per_block = 256;
int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
// 向上取整，确保所有元素都被处理
```

## 练习答案提示

<details>
<summary>点击查看提示（先尝试自己完成）</summary>

```cpp
// 1. 分配 device 内存
CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

// 2. 将数据从 host 复制到 device
CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

// 3. 启动 kernel
vector_add_kernel<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);

// 4. 将结果从 device 复制回 host
CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

// 5. 释放 device 内存
CUDA_CHECK(cudaFree(d_a));
CUDA_CHECK(cudaFree(d_b));
CUDA_CHECK(cudaFree(d_c));
```

</details>

## 常见问题

**Q: 为什么要分配 device 内存？**
A: GPU 有自己的独立显存，CPU 无法直接访问，必须使用 cudaMalloc 在 GPU 上分配内存。

**Q: cudaMemcpyHostToDevice 是什么意思？**
A: 这是拷贝方向枚举，表示数据从主机（CPU 内存）拷贝到设备（GPU 显存）。

**Q: 为什么 kernel 里需要边界检查？**
A: GPU 线程数通常是 block 大小的倍数，可能超过实际数据大小，需要防止越界访问。

**Q: cudaError_t 是什么？**
A: CUDA 错误类型，所有 CUDA API 都返回这个类型，应该检查错误。

## 性能提示

- 内存传输是性能瓶颈，尽量减少 Host-Device 数据传输
- 使用更大的数据量来 amortize 传输开销
- 考虑使用统一内存（Unified Memory）简化编程

## 下一步

完成本教程后，继续学习：
- [Lesson 02: 二维线程块和矩阵乘法](../02_matrix_mul/README.md)
