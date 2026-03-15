/**
 * Lesson 05: Streams 和 Events
 *
 * 目标：学习 CUDA 流和事件，实现并发执行
 *
 * TODO:
 * 1. 理解 CUDA stream 的概念
 * 2. 使用多个 stream 实现并发
 * 3. 使用 events 进行计时和同步
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 1000000
#define NUM_STREAMS 4

// 向量加法 kernel
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

// TODO: 实现使用多个 stream 的并发执行
void multi_stream_example(float* h_a, float* h_b, int n) {
    // TODO:
    // 1. 创建多个 streams (cudaStreamCreate)
    // 2. 为每个 stream 分配 device 内存
    // 3. 在每个 stream 中执行：H2D 拷贝 -> kernel -> D2H 拷贝
    // 4. 等待所有 streams 完成
}

int main() {
    printf("=== Lesson 05: Streams 和 Events ===\n\n");
    printDeviceinfo();

    int size = N * sizeof(float);
    int elements_per_stream = N / NUM_STREAMS;

    // 分配 host 内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 0.5f;
    }

    // 分配 device 内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // 配置 kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 测试 1: 默认 stream（顺序执行）
    printf("测试 1: 默认 stream (顺序执行)\n");
    printf("--------------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        // TODO: 在默认 stream 中执行 4 次向量加法

        timer.stopRecord();
        printf("时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 测试 2: 多个 streams（并发执行）
    printf("测试 2: 多个 streams (并发执行)\n");
    printf("--------------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        // TODO: 使用 4 个 streams 并发执行

        timer.stopRecord();
        printf("时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 测试 3: Events 计时
    printf("测试 3: Events 精确计时\n");
    printf("----------------------\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Kernel 执行时间 (Event): %.3f ms\n", elapsed);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 清理
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n程序完成!\n");

    return 0;
}
