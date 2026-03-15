/**
 * Lesson 06: CUDA Graph
 *
 * 目标：学习 CUDA Graph API，减少 kernel 启动开销
 *
 * TODO:
 * 1. 理解 CUDA Graph 的概念
 * 2. 创建和执行 CUDA Graph
 * 3. 比较 Graph 和传统启动方式的性能差异
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 10000
#define ITERATIONS 100

// 简单的向量加法 kernel
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 向量乘法 kernel
__global__ void vector_mul(float* a, float* b, float* c, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] * factor;
    }
}

// TODO: 使用 CUDA Graph API 重构下面的函数
void run_with_graph(float* d_a, float* d_b, float* d_c, int n) {
    // TODO:
    // 1. 创建 cudaGraph_t
    // 2. 添加 kernel nodes 到 graph
    // 3. 实例化 graph (cudaGraphInstantiate)
    // 4. 执行 graph (cudaGraphLaunch)
}

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

    // 测试 1: 传统方式（多次 kernel 启动）
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
        printf("总时间：%.3f ms (%d 次迭代)\n", timer.elapsedMs(), ITERATIONS);
        printf("平均每次迭代：%.3f ms\n\n", timer.elapsedMs() / ITERATIONS);
    }

    // 测试 2: CUDA Graph 方式
    printf("测试 2: CUDA Graph 方式\n");
    printf("----------------------\n");
    {
        // TODO: 实现 CUDA Graph 版本
        // 提示：参考 run_with_graph 函数

        printf("待实现...\n\n");
    }

    // 清理
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("程序完成!\n");

    return 0;
}
