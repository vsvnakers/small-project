# Lesson 06: CUDA Graph

## 学习目标

- 理解 CUDA Graph 的概念和优势
- 学习创建和执行 CUDA Graph
- 掌握 Graph 的更新和重用
- 了解适用场景

## 为什么需要 CUDA Graph？

### Kernel 启动开销

```cuda
// 传统方式：每次启动都有开销
for (int i = 0; i < 1000; i++) {
    kernel<<<blocks, threads>>>(...);  // ~5-10 µs 开销
}
```

### CUDA Graph 优势

```cuda
// 定义一次，执行多次
cudaGraphExec_t graph_exec;
// 创建 graph...

for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(graph_exec, 0);  // ~1-2 µs 开销
}
```

## CUDA Graph 基础

### Graph 组成

```
Graph
├── Kernel Node (kernel 执行)
├── Memcpy Node (内存拷贝)
├── Memset Node (内存设置)
├── Event Record Node
├── Event Wait Node
└── Empty Node (依赖控制)
```

### 基本流程

```
1. cudaGraphCreate 创建 graph
2. cudaGraphAddKernelNode 添加节点
3. cudaGraphInstantiate 创建可执行 graph
4. cudaGraphLaunch 执行
5. cudaGraphExecDestroy 清理
```

## 创建 Graph

### 示例：向量加法

```cuda
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 准备 kernel 参数
void* args[] = {&d_a, &d_b, &d_c, &n};

cudaKernelNodeParams kernel_params = {
    (void*)vector_add,           // kernel 函数
    args,                         // 参数
    nullptr,                      // extra (NULL for default)
    {blocks, 1, 1},              // grid dim
    {threads, 1, 1},             // block dim
    0,                            // shared mem size
    nullptr                       // kernel params (NULL for default)
};

// 添加 kernel node
cudaGraphNode_t node;
cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kernel_params);

// 实例化
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// 执行
cudaGraphLaunch(graph_exec, 0);
cudaDeviceSynchronize();

// 清理
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);
```

## 依赖关系

### 链式依赖

```cuda
cudaGraphNode_t node1, node2, node3;

// 添加第一个节点
cudaGraphAddKernelNode(&node1, graph, nullptr, 0, &params1);

// node2 依赖 node1
const cudaGraphNode_t deps1[] = {node1};
cudaGraphAddKernelNode(&node2, graph, deps1, 1, &params2);

// node3 依赖 node2
const cudaGraphNode_t deps2[] = {node2};
cudaGraphAddKernelNode(&node3, graph, deps2, 1, &params3);
```

### 并行节点

```cuda
cudaGraphNode_t node_a, node_b;

// 两个节点都无依赖
cudaGraphAddKernelNode(&node_a, graph, nullptr, 0, &params_a);
cudaGraphAddKernelNode(&node_b, graph, nullptr, 0, &params_b);

// 可以并发执行
```

## 添加 Memcpy Node

```cuda
cudaGraphNode_t memcpy_node;
cudaMemcpy3DParms memcpy_params = {
    {0},                                    // srcPos
    {h_src, 0, {size, 1, 1}, cudaHost},    // srcPtr
    {0},                                    // dstPos
    {d_dst, 0, {size, 1, 1}, cudaDevice},  // dstPtr
    {size, 1, 1},                          // extent
    cudaMemcpyHostToDevice                 // kind
};

cudaGraphAddMemcpyNode(&memcpy_node, graph, nullptr, 0, &memcpy_params);
```

## Graph 更新

### 更新参数

```cuda
// 首次创建
cudaGraphExec_t graph_exec = create_graph(params1);

// 更新参数后重用
cudaGraphNode_t* nodes = nullptr;
size_t num_nodes = 1;
cudaGraphGetNodes(graph, nodes, &num_nodes);

cudaKernelNodeParams new_params = {...};
cudaGraphExecKernelNodeSetParams(graph_exec, node, &new_params);

// 直接执行，无需重新创建
cudaGraphLaunch(graph_exec, 0);
```

## 实战：迭代算法

### 多轮迭代

```cuda
struct GraphParams {
    float* d_data;
    int n;
    int blocks;
    int threads;
};

cudaGraphExec_t create_iteration_graph(GraphParams& params, int iterations) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t* nodes = (cudaGraphNode_t*)malloc(iterations * sizeof(cudaGraphNode_t));

    for (int i = 0; i < iterations; i++) {
        void* args[] = {&params.d_data, &params.n};

        cudaKernelNodeParams kernel_params = {
            (void*)update_kernel,
            args,
            nullptr,
            {params.blocks, 1, 1},
            {params.threads, 1, 1},
            0, nullptr
        };

        if (i == 0) {
            cudaGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &kernel_params);
        } else {
            cudaGraphAddKernelNode(&nodes[i], graph, &nodes[i-1], 1, &kernel_params);
        }
    }

    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    free(nodes);

    return graph_exec;
}
```

## 性能对比

### 启动开销

| 方式 | 开销 | 适用场景 |
|------|------|----------|
| 传统启动 | 5-10 µs | 偶尔执行 |
| CUDA Graph | 1-2 µs | 频繁执行 |

### 示例对比

```cuda
int iterations = 1000;

// 传统方式
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < iterations; i++) {
    kernel<<<blocks, threads>>>(...);
}
cudaDeviceSynchronize();
// 总时间：~8 ms

// CUDA Graph
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graph_exec, 0);
}
cudaDeviceSynchronize();
// 总时间：~2 ms
```

## 练习

### TODO: 完成 cuda_graph.cu

1. 创建包含两个 kernel 的 graph
2. 设置依赖关系
3. 比较与传统方式的性能差异

### 进阶练习

1. 实现多轮迭代的 graph
2. 添加 memcpy node
3. 使用 graph update API

## 调试 Graph

### 导出 Graph 可视化

```cuda
// 导出为 DOT 文件
cudaGraphDebugDotPrint(graph, "graph.dot", 0);

// 用 graphviz 查看
dot -Tpng graph.dot -o graph.png
```

### 检查错误

```cuda
cudaError_t err = cudaGraphInstantiate(&graph_exec, graph,
                                        &first_error_node,
                                        &log_buffer, &log_buffer_size);
if (err != cudaSuccess) {
    printf("Graph 实例化失败：%s\n", log_buffer);
}
```

## 适用场景

### 适合使用 Graph

- ✅ 小 kernel 频繁执行
- ✅ 固定模式的迭代算法
- ✅ 需要精确控制执行顺序
- ✅ 深度学习推理

### 不适合使用 Graph

- ❌ 只执行一次的大 kernel
- ❌ 动态变化的执行模式
- ❌ 需要 CPU-GPU 频繁交互

## 常见问题

### Q1: Graph 可以包含条件分支吗？

目前不支持。Graph 是静态的，执行路径固定。

### Q2: 如何调试 Graph？

使用 `cudaGraphDebugDotPrint` 导出可视化。

### Q3: Graph 可以嵌套吗？

可以，支持 child graph。

## 下一步

完成 [cuda_graph.cu](../tutorials/06_cuda_graph/cuda_graph.cu) 的 TODO 部分，然后学习 [Lesson 07: Tensor Core](07_tensor_core.md)。
