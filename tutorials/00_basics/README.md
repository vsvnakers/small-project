# Lesson 00: CUDA 基础 - Hello World

本教程介绍 CUDA 程序的基本结构和执行模型。

## 学习目标

- 理解 CUDA 程序的基本结构
- 了解 Host（CPU）和 Device（GPU）的区别
- 掌握 Kernel 函数的定义和调用方式
- 理解线程层次结构（Thread/Block/Grid）

## 文件结构

```
00_basics/
├── hello_world.cu          # TODO 练习代码（你需要补全）
├── solution/
│   └── hello_world.cu      # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

在项目根目录构建整个项目：

```bash
cd <project-root>
mkdir -p build && cd build
cmake ..
make -j
```

其中 `<project-root>` 是你克隆项目后的根目录。

编译完成后，可执行文件位于：
- `tutorials/00_basics/hello_world` - TODO 版本
- `tutorials/00_basics/hello_world_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
# 编译 TODO 版本
nvcc -o hello_world hello_world.cu

# 编译参考答案
nvcc -o hello_world_solution solution/hello_world.cu
```

## 运行方法

```bash
# 运行 TODO 版本（补全代码后）
./hello_world

# 运行参考答案
./hello_world_solution
```

### 预期输出

```
启动 CUDA kernel...
配置：4 blocks x 4 threads = 16 线程
Hello from thread 0
Hello from thread 1
Hello from thread 2
Hello from thread 3
Hello from thread 4
Hello from thread 5
Hello from thread 6
Hello from thread 7
Hello from thread 8
Hello from thread 9
Hello from thread 10
Hello from thread 11
Hello from thread 12
Hello from thread 13
Hello from thread 14
Hello from thread 15
CUDA 程序完成!
```

> **注意：线程输出顺序可能不固定**，因为 GPU 线程是并行执行的。
>
> **如果在某些系统（如 WSL2）上看不到 "Hello from thread X" 输出**，这是正常的。
> 某些驱动配置会缓冲 CUDA kernel 中的 printf 输出，这不影响 CUDA 功能。
> 你可以通过其他方式验证 kernel 是否正常执行（如查看 GPU 利用率、使用 profiling 工具等）。

## 练习任务

1. **补全 kernel 函数**：计算全局线程 ID 并打印
2. **修改配置**：尝试不同的 block 和 thread 数量
3. **理解执行**：观察输出，理解并行执行的特点

## 关键概念

### Kernel 函数定义

```cpp
__global__ void hello_kernel() {
    // __global__ 表示这是 kernel 函数
    // 在 GPU 上执行，由 CPU 调用
}
```

### Kernel 启动

```cpp
kernel_name<<<grid_size, block_size>>>(arguments);
```

- `grid_size`：Grid 中的 Block 数量
- `block_size`：每个 Block 中的 Thread 数量

### 线程索引计算

```cpp
int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x`：当前 Block 在 Grid 中的索引
- `blockDim.x`：每个 Block 的线程数
- `threadIdx.x`：当前线程在 Block 中的索引

## 常见问题

**Q: 为什么输出顺序不固定？**
A: GPU 线程是并行执行的，没有固定的执行顺序。

**Q: blockDim 和 blockIdx 有什么区别？**
A: `blockDim` 是每个 block 的大小（固定值），`blockIdx` 是当前 block 的索引（变化值）。

**Q: 如何增加线程数量？**
A: 增加 `number_of_blocks` 或 `threads_per_block` 的值。

## 下一步

完成本教程后，继续学习：
- [Lesson 01: 内存管理](../01_vector_add/README.md)
