# Lesson 03: 共享内存

本教程深入学习共享内存的使用和 Bank Conflict 优化。

## 学习目标

- 掌握共享内存的声明和使用
- 理解树形归约算法
- 学习 Bank Conflict 的概念和影响
- 了解 `__syncthreads()` 同步机制

## 文件结构

```
03_shared_memory/
├── shared_mem.cu           # TODO 练习代码
├── solution/
│   └── shared_mem.cu       # 参考答案
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
- `tutorials/03_shared_memory/shared_mem` - TODO 版本
- `tutorials/03_shared_memory/shared_mem_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/03_shared_memory

# 编译 TODO 版本
nvcc -o shared_mem shared_mem.cu -I../../include

# 编译参考答案
nvcc -o shared_mem_solution solution/shared_mem.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./shared_mem

# 运行参考答案
./shared_mem_solution
```

### 预期输出

```
=== Lesson 03: 共享内存 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

数组大小：1048576
期望结果：1048576

基础版本：X.XXX ms
  结果：1048576 (误差：0)

共享内存版本：X.XXX ms
  结果：1048576 (误差：0)
```

## 练习任务

1. **声明共享内存**：使用 `__shared__` 关键字
2. **数据加载**：将全局内存数据加载到共享内存
3. **树形归约**：实现并行归约算法
4. **同步线程**：正确使用 `__syncthreads()`

## 关键概念

### 共享内存声明

```cpp
// 静态声明（编译时确定大小）
__shared__ float sdata[BLOCK_SIZE];

// 动态声明（运行时确定大小）
// Kernel 声明：
extern __shared__ float sdata[];
// Kernel 调用：
kernel<<<blocks, threads, shared_mem_size>>>(...);
```

### 树形归约算法

```cpp
// 并行归约：每步将问题规模减半
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();  // 确保所有线程完成后再继续
}
```

### 归约过程示意图

```
初始：[1, 2, 3, 4, 5, 6, 7, 8]

第 1 步：[1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
第 2 步：[3+7, 11+15] = [10, 26]
第 3 步：[10+26] = [36]
```

### 同步机制

```cpp
// Block 内同步：等待所有线程到达该点
__syncthreads();

// 注意：只能同步同一 block 内的线程
```

## Bank Conflict 详解

### 什么是 Bank Conflict？

共享内存被划分为 32 个等宽的内存模块（Banks）。当多个线程同时访问同一 Bank 的不同地址时，会发生 Bank Conflict，导致串行化访问。

### 避免 Bank Conflict

```cpp
// 可能有 bank conflict 的访问（步长为 2 的幂）
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    sdata[tid] += sdata[tid + s];  // 冲突！
}

// 无 bank conflict 的访问（使用偏移）
__shared__ float sdata[BLOCK_SIZE + 1];  // 添加填充
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    sdata[tid] += sdata[tid + s];  // 无冲突
}
```

## 练习答案提示

<details>
<summary>共享内存归约答案</summary>

```cpp
__global__ void reduce_shared(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 声明共享内存
    __shared__ float sdata[BLOCK_SIZE];

    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 将局部和存入共享内存
    sdata[tid] = sum;
    __syncthreads();

    // 树形归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将结果写入输出
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

</details>

## 常见问题

**Q: 共享内存和全局内存有什么区别？**
A: 共享内存位于 GPU 芯片上，速度比全局内存快约 100 倍，但只在 block 内共享。

**Q: 为什么需要__syncthreads()？**
A: 确保 block 内所有线程都完成数据加载后，再进行归约计算，避免竞态条件。

**Q: Bank Conflict 会如何影响性能？**
A: 发生 Bank Conflict 时，访问会串行化，可能降低 32 倍性能。

**Q: 共享内存有多大？**
A: 通常每个 SM 有 48-96 KB 共享内存，具体取决于 GPU 架构。

## 性能提示

1. **最大化共享内存使用**：减少全局内存访问
2. **避免 Bank Conflict**：设计合理的访问模式
3. **使用合适的 Block 大小**：通常是 32 的倍数
4. **减少同步次数**：在满足正确性的前提下最小化 `__syncthreads()`

## 下一步

完成本教程后，继续学习：
- [Lesson 04: 原子操作](../04_atomic_ops/README.md)
