/**
 * Lesson 06: CUDA Graph (参考答案)
 *
 * 编译：nvcc -o cuda_graph_solution cuda_graph.cu
 * 运行：./cuda_graph_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 10000
#define ITERATIONS 100

// ============================================================================
// Kernel 定义
// ============================================================================

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_mul(float* a, float* b, float* c, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] * factor;
    }
}

// ============================================================================
// CUDA Graph 参数结构
// ============================================================================

struct GraphParams {
    float *d_a;
    float *d_b;
    float *d_c;
    int n;
    int blocks;
    int threads;
};

// ============================================================================
// 创建 CUDA Graph
// ============================================================================

cudaGraphExec_t create_graph(GraphParams& params) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // 准备 kernel 参数
    void* add_args[] = {&params.d_a, &params.d_b, &params.d_c, &params.n};
    float mul_factor = 1.0f;
    void* mul_args[] = {&params.d_a, &params.d_b, &params.d_c, &params.n, &mul_factor};

    // 创建 kernel node
    cudaGraphNode_t add_node, mul_node;

    cudaKernelNodeParams add_params;
    add_params.func = (void*)vector_add;
    add_params.kernelParams = (void**)add_args;
    add_params.extra = nullptr;
    add_params.gridDim = dim3{params.blocks, 1, 1};
    add_params.blockDim = dim3{params.threads, 1, 1};
    add_params.sharedMemBytes = 0;

    cudaKernelNodeParams mul_params;
    mul_params.func = (void*)vector_mul;
    mul_params.kernelParams = (void**)mul_args;
    mul_params.extra = nullptr;
    mul_params.gridDim = dim3{params.blocks, 1, 1};
    mul_params.blockDim = dim3{params.threads, 1, 1};
    mul_params.sharedMemBytes = 0;

    // 添加 nodes 到 graph（mul 依赖 add）
    CUDA_CHECK(cudaGraphAddKernelNode(&add_node, graph, nullptr, 0, &add_params));

    const cudaGraphNode_t dependencies[] = {add_node};
    CUDA_CHECK(cudaGraphAddKernelNode(&mul_node, graph, dependencies, 1, &mul_params));

    // 实例化 graph
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // 清理
    cudaGraphDestroy(graph);

    return graph_exec;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 06: CUDA Graph ===\n\n");
    printDeviceinfo();

    int size = N * sizeof(float);

    // 分配内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 0.5f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 设置 graph 参数
    GraphParams params = {d_a, d_b, d_c, N, blocks, threads};

    // 创建 graph
    cudaGraphExec_t graph_exec = create_graph(params);
    printf("CUDA Graph 创建成功!\n\n");

    // -------------------------------------------------------------------------
    // 测试 1: 传统方式（多次 kernel 启动）
    // -------------------------------------------------------------------------
    printf("测试 1: 传统 Kernel 启动方式\n");
    printf("--------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        for (int i = 0; i < ITERATIONS; i++) {
            vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
            vector_mul<<<blocks, threads>>>(d_a, d_b, d_c, N, 1.0f);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stopRecord();
        float total = timer.elapsedMs();
        printf("总时间：%.3f ms (%d 次迭代)\n", total, ITERATIONS);
        printf("平均每次迭代：%.3f ms\n", total / ITERATIONS);
        printf("每次 kernel 启动开销：~%.3f µs\n\n", (total / ITERATIONS / 2) * 1000);
    }

    // -------------------------------------------------------------------------
    // 测试 2: CUDA Graph 方式
    // -------------------------------------------------------------------------
    printf("测试 2: CUDA Graph 方式\n");
    printf("----------------------\n");
    {
        // 预热
        CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        SimpleTimer timer;
        timer.startRecord();

        for (int i = 0; i < ITERATIONS; i++) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stopRecord();
        float total = timer.elapsedMs();
        printf("总时间：%.3f ms (%d 次迭代)\n", total, ITERATIONS);
        printf("平均每次迭代：%.3f ms\n", total / ITERATIONS);
        printf("Graph 启动开销：~%.3f µs\n\n", (total / ITERATIONS) * 1000);
    }

    // -------------------------------------------------------------------------
    // 测试 3: 使用 cudaGraphExecUpdate 更新参数后执行
    // -------------------------------------------------------------------------
    printf("测试 3: Graph 参数更新\n");
    printf("----------------------\n");

    // 修改参数
    params.n = N / 2;
    int new_blocks = (params.n + threads - 1) / threads;
    params.blocks = new_blocks;

    // 重新创建 graph
    cudaGraphExec_t new_graph_exec = create_graph(params);

    {
        SimpleTimer timer;
        timer.startRecord();

        for (int i = 0; i < ITERATIONS; i++) {
            CUDA_CHECK(cudaGraphLaunch(new_graph_exec, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stopRecord();
        printf("N = %d, 总时间：%.3f ms\n", params.n, timer.elapsedMs());
    }

    // 清理 graph
    cudaGraphExecDestroy(graph_exec);
    cudaGraphExecDestroy(new_graph_exec);

    // 清理内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n=== 知识点总结 ===\n");
    printf("1. CUDA Graph 将多个操作预定义为图，减少启动开销\n");
    printf("2. 适合频繁执行相同操作序列的场景\n");
    printf("3. Graph 可以包含 kernel、内存拷贝、事件等节点\n");
    printf("4. 使用 cudaGraphInstantiate 创建可执行的 graph\n");
    printf("5. 典型加速：从 5-10µs 降到 1-2µs 启动开销\n");

    printf("\n程序完成!\n");

    return 0;
}
