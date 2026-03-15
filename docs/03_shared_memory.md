# Lesson 03: 共享内存

## 学习目标

- 理解共享内存的工作原理
- 学习使用共享内存优化 kernel
- 了解 Bank Conflict 及避免方法
- 实现归约操作

## 共享内存概述

### 什么是共享内存？

共享内存是 GPU 上的一种**片上内存**，由同一个 block 内的所有线程共享。

```
┌─────────────────────────────────┐
│        Streaming Multiprocessor │
│  ┌───────────────────────────┐  │
│  │    Shared Memory (64KB)   │  │ ← 片上内存
│  │  ┌─────┬─────┬─────┐      │  │
│  │  │Block│Block│Block│      │  │
│  │  │  0  │  1  │  2  │      │  │
│  │  └─────┴─────┴─────┘      │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

### 性能对比

| 内存类型 | 带宽 | 延迟 |
|----------|------|------|
| 共享内存 | ~10 TB/s | ~100 cycles |
| L1 缓存 | ~1 TB/s | ~200 cycles |
| 全局内存 | ~1 TB/s | ~500 cycles |

## 基本语法

### 声明共享内存

```cuda
__global__ void kernel() {
    // 静态声明（编译时确定大小）
    __shared__ float sdata[256];

    // 使用
    sdata[threadIdx.x] = ...;
}
```

### 动态共享内存

```cuda
// 声明 extern 共享内存
__global__ void kernel(...) {
    extern __shared__ float sdata[];
    // 使用
}

// 启动时指定大小
kernel<<<blocks, threads, shared_mem_size>>>(...);
```

## 归约操作示例

### 树形归约

```cuda
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // 树形归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### 执行流程

```
初始：[1, 2, 3, 4, 5, 6, 7, 8]
Round 1: [3, X, 7, X, 11, X, 15, X]
Round 2: [10, X, X, X, 26, X, X, X]
Round 3: [36, X, X, X, X, X, X, X]
```

## Bank Conflict

### 什么是 Bank Conflict？

共享内存被分为 32 个 bank，每个 bank 宽度为 4 字节。

```
Bank 0  Bank 1  Bank 2  ...  Bank 31
┌────┐ ┌────┐ ┌────┐      ┌────┐
│  0 │ │  1 │ │  2 │  ... │ 31 │
├────┤ ├────┤ ├────┤      ├────┤
│ 32 │ │ 33 │ │ 34 │  ... │ 63 │
├────┤ ├────┤ ├────┤      ├────┤
│ 64 │ │ 65 │ │ 66 │  ... │ 95 │
└────┘ └────┘ └────┘      └────┘
```

### Conflict 情况

```cuda
// 32-way conflict! 所有线程访问同一个 bank
for (int i = 0; i < 32; i++) {
    sdata[i * 32] = ...;  // 都访问 bank 0
}
```

### 避免方法

```cuda
// 添加 padding
__shared__ float sdata[256 + 8];  // 避免 32 的倍数

// 或者重排索引
int idx = tid + (tid / 32);  // 跳过某些位置
```

## 优化技巧

### 1. 无分支归约

```cuda
// 避免 warp divergence
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

### 2. Warp 级优化

```cuda
// 最后 32 个线程不需要 __syncthreads()
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
// warp 内隐式同步
if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

### 3. 向量化加载

```cuda
// 使用 float4 加载
float4 in = reinterpret_cast<const float4*>(&input[idx])[0];
sdata[tid * 4 + 0] = in.x;
sdata[tid * 4 + 1] = in.y;
sdata[tid * 4 + 2] = in.z;
sdata[tid * 4 + 3] = in.w;
```

## 实战：直方图

### 使用共享内存优化

```cuda
__global__ void histogram(unsigned char* input, int* hist, int n) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;

    // 初始化共享内存
    for (int i = tid; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // 在共享内存中累积
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        atomicAdd(&local_hist[input[idx]], 1);
    }
    __syncthreads();

    // 合并到全局内存
    for (int i = tid; i < 256; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&hist[i], local_hist[i]);
        }
    }
}
```

## 练习

### TODO: 完成 shared_mem.cu

1. 实现树形归约
2. 使用 `__syncthreads()` 正确同步
3. 验证结果正确性

### 进阶练习

1. 测量共享内存和全局内存的性能差异
2. 实现无 bank conflict 版本
3. 比较不同 block size 的影响

## 调试共享内存

### 检查共享内存使用

```cuda
cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, kernel);
printf("Shared memory: %zu bytes\n", attr.sharedSizeBytes);
```

### 检测 Bank Conflict

使用 Nsight Compute:
```bash
ncu --metrics sm__shared_memory_efficiency ./your_program
```

## 常见问题

### Q1: __syncthreads() 可以省略吗？

不可以！以下情况必须使用：

```cuda
sdata[tid] = ...;     // 写入
__syncthreads();      // 必须！
result = sdata[...];  // 读取
```

### Q2: 共享内存大小限制？

- Volta+: 64 KB/SM
- Pascal: 48 KB/SM

每个 block 最多可使用 64 KB。

### Q3: 如何最大化利用？

```cuda
//  occupancy 计算
int block_size = 256;
int shared_mem = 8 * 1024;  // 8 KB
int active_blocks = min(32, 64 * 1024 / shared_mem);
```

## 下一步

完成 [shared_mem.cu](../tutorials/03_shared_memory/shared_mem.cu) 的 TODO 部分，然后学习 [Lesson 04: 原子操作](04_atomic_operations.md)。
