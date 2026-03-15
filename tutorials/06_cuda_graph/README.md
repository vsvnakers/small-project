# Lesson 06: CUDA Graph

本教程介绍 CUDA Graph API，用于减少 kernel 启动开销。

## 学习目标

- 理解 CUDA Graph 的概念和优势
- 掌握创建和执行 CUDA Graph 的流程
- 学习 Kernel Node 的添加和依赖设置
- 比较 Graph 和传统启动方式的性能差异

## 文件结构

```
06_cuda_graph/
├── cuda_graph.cu           # TODO 练习代码
├── solution/
│   └── cuda_graph.cu       # 参考答案
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
- `tutorials/06_cuda_graph/cuda_graph` - TODO 版本
- `tutorials/06_cuda_graph/cuda_graph_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/06_cuda_graph

# 编译 TODO 版本
nvcc -o cuda_graph cuda_graph.cu -I../../include

# 编译参考答案
nvcc -o cuda_graph_solution solution/cuda_graph.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./cuda_graph

# 运行参考答案
./cuda_graph_solution
```

### 预期输出

```
=== Lesson 06: CUDA Graph ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

测试 1: 传统 Kernel 启动方式
--------------------------
总时间：XX.XXX ms (100 次迭代)
平均每次迭代：X.XXX ms
每次 kernel 启动开销：~X.XX µs

测试 2: CUDA Graph 方式
----------------------
总时间：XX.XXX ms (100 次迭代)
平均每次迭代：X.XX ms
Graph 启动开销：~X.XX µs
```

## 练习任务

1. **创建 CUDA Graph**：使用 `cudaGraphCreate`
2. **添加 Kernel Nodes**：使用 `cudaGraphAddKernelNode`
3. **设置依赖关系**：定义 node 之间的执行顺序
4. **实例化和执行**：使用 `cudaGraphInstantiate` 和 `cudaGraphLaunch`

## 关键概念

### 什么是 CUDA Graph？

CUDA Graph 是一种将一系列 CUDA 操作（kernel、内存拷贝、事件等）预定义为有向无环图（DAG）的机制。Graph 可以重复执行，显著减少 CPU 启动开销。

### Graph 执行流程

```
1. 创建 Graph (cudaGraphCreate)
       ↓
2. 添加 Nodes (cudaGraphAddKernelNode 等)
       ↓
3. 实例化 Graph (cudaGraphInstantiate)
       ↓
4. 执行 Graph (cudaGraphLaunch)
       ↓
5. 销毁 Graph (cudaGraphExecDestroy)
```

### Graph 优势

| 传统方式 | CUDA Graph |
|----------|------------|
| 每次启动都有 CPU 开销 | 一次定义，多次执行 |
| 不适合频繁的小 kernel | 大幅减少启动开销 |
| 难以优化依赖关系 | 明确定义依赖关系 |

### 核心 API

```cpp
// 1. 创建 Graph
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 2. 定义 Kernel Node 参数
cudaKernelNodeParams kernel_params;
kernel_params.func = (void*)my_kernel;
kernel_params.kernelParams = (void**)args;
kernel_params.gridDim = dim3{blocks, 1, 1};
kernel_params.blockDim = dim3{threads, 1, 1};
kernel_params.sharedMemBytes = 0;

// 3. 添加 Kernel Node
cudaGraphNode_t node;
cudaGraphAddKernelNode(&node, graph, dependencies, num_deps, &kernel_params);

// 4. 实例化 Graph
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// 5. 执行 Graph
cudaGraphLaunch(graph_exec, stream);

// 6. 清理
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);
```

## 练习答案提示

<details>
<summary>创建和执行 CUDA Graph 的完整流程</summary>

```cpp
// 准备参数
void* add_args[] = {&d_a, &d_b, &d_c, &n};
float mul_factor = 1.0f;
void* mul_args[] = {&d_a, &d_b, &d_c, &n, &mul_factor};

// 1. 创建 Graph
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 2. 定义 Kernel 参数
cudaKernelNodeParams add_params;
add_params.func = (void*)vector_add;
add_params.kernelParams = (void**)add_args;
add_params.gridDim = dim3{blocks, 1, 1};
add_params.blockDim = dim3{threads, 1, 1};
add_params.sharedMemBytes = 0;

cudaKernelNodeParams mul_params;
mul_params.func = (void*)vector_mul;
mul_params.kernelParams = (void**)mul_args;
mul_params.gridDim = dim3{blocks, 1, 1};
mul_params.blockDim = dim3{threads, 1, 1};
mul_params.sharedMemBytes = 0;

// 3. 添加 Nodes（mul 依赖 add）
cudaGraphNode_t add_node, mul_node;
cudaGraphAddKernelNode(&add_node, graph, nullptr, 0, &add_params);

const cudaGraphNode_t dependencies[] = {add_node};
cudaGraphAddKernelNode(&mul_node, graph, dependencies, 1, &mul_params);

// 4. 实例化 Graph
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// 5. 执行 Graph（可重复执行）
for (int i = 0; i < ITERATIONS; i++) {
    cudaGraphLaunch(graph_exec, 0);
}
cudaDeviceSynchronize();

// 6. 清理
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);
```

</details>

## 常见问题

**Q: CUDA Graph 适合什么场景？**
A: 适合频繁执行相同操作序列的场景，特别是小 kernel 或延迟敏感的应用。

**Q: Graph 可以包含哪些类型的 Node？**
A: Kernel、内存拷贝、事件、memcpy/memset、CPU 回调等。

**Q: 如何更新 Graph 的参数？**
A: 使用 `cudaGraphExecUpdate` 可以在不重新创建 Graph 的情况下更新参数。

**Q: Graph 的性能提升有多少？**
A: 通常可以将启动开销从 5-10µs 降到 1-2µs，对于小 kernel 提升更明显。

## 性能提示

1. **减少重复创建**：Graph 应该创建一次，重复执行多次
2. **合理设置依赖**：只设置必要的依赖关系，最大化并发
3. **预热 Graph**：首次执行前至少启动一次 Graph 进行预热

## 下一步

完成本教程后，继续学习：
- [Lesson 07: Tensor Core](../07_tensor_core/README.md)
