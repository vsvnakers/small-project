# Lesson 04: 原子操作

本教程介绍 CUDA 原子操作和线程同步技术。

## 学习目标

- 理解原子操作的必要性
- 掌握 `atomicAdd` 等原子 API
- 学习实现原子计数器
- 实现原子直方图

## 文件结构

```
04_atomic_ops/
├── atomic_ops.cu           # TODO 练习代码
├── solution/
│   └── atomic_ops.cu       # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

```bash
cd <project-root>
mkdir -p build && cd build
cmake ..
make -j
```

其中 `<project-root>` 是你克隆项目后的根目录。

可执行文件位于：
- `tutorials/04_atomic_ops/atomic_ops` - TODO 版本
- `tutorials/04_atomic_ops/atomic_ops_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd tutorials/04_atomic_ops

# 编译 TODO 版本
nvcc -o atomic_ops atomic_ops.cu -I../../include

# 编译参考答案
nvcc -o atomic_ops_solution solution/atomic_ops.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./atomic_ops

# 运行参考答案
./atomic_ops_solution
```

### 预期输出

```
=== Lesson 04: 原子操作 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

测试 1: 原子计数器
------------------
线程总数：1000000
计数器结果：1000000 (期望：1000000)

测试 2: 原子直方图
------------------
GPU 时间：X.XXX ms
CPU 时间：XX.XXX ms

结果验证 (前 10 个 bucket):
  Bucket 0: GPU=3906, CPU=3906 ✓
  Bucket 1: GPU=3906, CPU=3906 ✓
  ...
```

## 练习任务

1. **实现原子计数器**：使用 `atomicAdd` 实现全局计数
2. **实现原子直方图**：统计数据分布
3. **对比性能**：比较 GPU 和 CPU 版本的执行时间

## 关键概念

### 什么是原子操作？

原子操作是不可中断的操作。当多个线程同时访问同一内存位置时，原子操作确保它们按顺序执行，避免竞态条件。

### 常用原子 API

```cpp
// 原子加
int atomicAdd(int* address, int val);

// 原子减
int atomicSub(int* address, int val);

// 原子交换
int atomicExch(int* address, int val);

// 原子最大值
int atomicMax(int* address, int val);

// 原子最小值
int atomicMin(int* address, int val);

// 原子与
int atomicAnd(int* address, int val);

// 原子或
int atomicOr(int* address, int val);

// 原子异或
int atomicXor(int* address, int val);

// 原子 CAS (Compare-And-Swap)
int atomicCAS(int* address, int compare, int val);
```

### 原子计数器实现

```cpp
__global__ void atomic_counter_basic(int* counter, int n) {
    // 每个线程都对计数器加 1
    atomicAdd(counter, 1);
}
```

### 原子直方图实现

```cpp
__global__ void atomic_histogram(unsigned char* input, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned char value = input[idx];
        // 原子地增加对应 bucket 的计数
        atomicAdd(&histogram[value], 1);
    }
}
```

## 练习答案提示

<details>
<summary>原子计数器答案</summary>

```cpp
__global__ void atomic_counter_basic(int* counter, int n) {
    // 每个线程都对计数器加 1
    // atomicAdd 返回加之前的值，但这里我们不需要
    atomicAdd(counter, 1);
}
```

</details>

<details>
<summary>原子直方图答案</summary>

```cpp
__global__ void atomic_histogram(unsigned char* input, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned char value = input[idx];
        // 原子地增加对应 bucket 的计数
        atomicAdd(&histogram[value], 1);
    }
}
```

</details>

## 常见问题

**Q: 为什么需要原子操作？**
A: 当多个线程同时更新同一内存位置时，非原子操作会导致数据竞争，产生错误结果。

**Q: 原子操作的性能如何？**
A: 原子操作比非原子操作慢，因为它们需要串行化。但在处理共享数据时是必需的。

**Q: 原子操作支持哪些数据类型？**
A: 支持 `int`、`unsigned int`、`unsigned long long`、`float`、`double`（计算能力 6.0+）等。

**Q: 如何减少原子操作的性能开销？**
A:
1. 使用共享内存进行 block 内归约，最后再原子地写入全局内存
2. 使用原子操作较少的算法设计
3. 利用线程局部存储

## 性能提示

1. **最小化原子冲突**：设计算法减少线程对同一地址的访问
2. **使用共享内存**：先在共享内存归约，再原子地写入全局内存
3. **选择合适的数据结构**：如每个线程有自己的计数器，最后再合并

## 下一步

完成本教程后，继续学习：
- [Lesson 05: CUDA Streams 和 Events](../05_streams/README.md)
