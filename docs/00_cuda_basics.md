# Lesson 00: CUDA 基础与环境设置

## 学习目标

- 了解 GPU 架构和 CUDA 编程模型
- 设置 CUDA 开发环境
- 编写第一个 CUDA 程序
- 理解 kernel 启动配置

## GPU 架构简介

### CPU vs GPU

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 少 (4-64) | 多 (1000+) |
| 设计目标 | 低延迟 | 高吞吐 |
| 适用场景 | 串行任务 | 并行任务 |

### NVIDIA GPU 架构演进

```
Volta (V100)   → 首次引入 Tensor Core
Turing (RTX20) → RT Core, 改进的 Tensor Core
Ampere (RTX30) → 第二代 Tensor Core, TF32 精度
Ada (RTX40)    → 第三代 Tensor Core, FP8 精度
```

## CUDA 编程模型

### 执行模型

```
Host (CPU)
    ↓ 启动
Kernel (GPU)
    ↓
Grid
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   └── ...
├── Block (1,0)
│   └── ...
└── ...
```

### 关键概念

- **Host**: CPU 及其内存
- **Device**: GPU 及其内存
- **Kernel**: 在 GPU 上执行的函数
- **Thread**: 单个执行单元
- **Block**: 线程块，共享内存
- **Grid**: 线程块网格

## 第一个 CUDA 程序

### Hello World Kernel

```cuda
__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", idx);
}

int main() {
    // 启动 4 个 block，每个 block 4 个线程
    hello_kernel<<<4, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 代码解析

```cuda
// __global__ 表示 kernel 函数
__global__ void hello_kernel() {
    // blockIdx.x: block 在 grid 中的索引
    // blockDim.x: 每个 block 的线程数
    // threadIdx.x: 线程在 block 中的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// <<<grid_size, block_size>>>
hello_kernel<<<4, 4>>>();
```

## 线程索引计算

### 一维索引

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二维索引

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

### 三维索引

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

## 内存管理

### 基本 API

```cuda
// 分配 device 内存
cudaMalloc(&d_ptr, size);

// 内存拷贝
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

// 释放 device 内存
cudaFree(d_ptr);
```

### 内存拷贝方向

| 常量 | 方向 |
|------|------|
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU |

## 编译和运行

### 编译命令

```bash
nvcc -o hello_world hello_world.cu
```

### 运行并检查输出

```bash
./hello_world
```

### 使用 cuda-gdb 调试

```bash
nvcc -g -G -o hello_world hello_world.cu
cuda-gdb ./hello_world
```

## 练习

### TODO: 完成 hello_world.cu

1. 补全 kernel 函数，让每个线程打印自己的线程 ID
2. 尝试修改 block 和 grid 的大小
3. 观察输出结果的变化

### 思考题

1. 如果设置 `<<<10, 10>>>`，总共有多少个线程？
2. 线程 ID 的范围是多少？
3. 线程执行顺序是确定的吗？

## 常见错误

### 忘记同步

```cuda
// 错误：没有等待 GPU 完成
kernel<<<blocks, threads>>>();
// 立即访问结果会出错

// 正确：
kernel<<<blocks, threads>>>();
cudaDeviceSynchronize();
```

### 内存越界

```cuda
// 确保检查边界
__global__ void kernel(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // 边界检查！
        arr[idx] = ...;
    }
}
```

## 下一步

完成 [hello_world.cu](../tutorials/00_basics/hello_world.cu) 的 TODO 部分，然后与 [solution/hello_world.cu](../tutorials/00_basics/solution/hello_world.cu) 对比。

准备好后，继续学习 [Lesson 01: 内存模型](01_memory_model.md)。
