/**
 * Lesson 09: 性能优化
 *
 * 目标：学习 CUDA 性能优化技巧
 *
 * TODO:
 * 1. 理解内存合并访问
 * 2. 优化 occupancy
 * 3. 使用 profiler 分析性能瓶颈
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N (1024 * 1024 * 16)  // 16M 元素

// 未优化的 kernel：随机内存访问
__global__ void naive_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 随机访问模式（未合并）
        int random_idx = (idx * 7919) % n;  // 质数乘法制造"随机"访问
        output[idx] = input[random_idx] * 2.0f + 1.0f;
    }
}

// TODO: 优化内存访问模式
__global__ void optimized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: 改为顺序访问（合并访问）
        // output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// 使用共享内存优化
__global__ void shared_mem_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[256];

    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // 处理共享内存中的数据
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f + 1.0f;
    }
}

// CPU 版本
void process_cpu(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Lesson 09: 性能优化 ===\n\n");
    printDeviceinfo();

    int size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("数据大小：%s\n", formatBytes(size));
    printf("线程配置：%d blocks x %d threads\n\n", blocks, threads);

    // 测试 1: 未优化版本
    printf("测试 1: 未优化 (随机访问)\n");
    printf("------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();
        naive_kernel<<<blocks, threads>>>(d_input, d_output, N);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("时间：%.3f ms\n", timer.elapsedMs());

        float bandwidth = (size * 2.0) / (timer.elapsedMs() * 1e-6) / 1e9;
        printf("带宽：%.2f GB/s\n\n", bandwidth);
    }

    // 测试 2: 优化版本
    printf("测试 2: 优化 (顺序访问)\n");
    printf("----------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        // TODO: 调用 optimized_kernel

        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("时间：待实现\n");
    }

    // 测试 3: 共享内存版本
    printf("\n测试 3: 共享内存优化\n");
    printf("------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();
        shared_mem_kernel<<<blocks, threads>>>(d_input, d_output, N);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("时间：%.3f ms\n", timer.elapsedMs());

        float bandwidth = (size * 2.0) / (timer.elapsedMs() * 1e-6) / 1e9;
        printf("带宽：%.2f GB/s\n", bandwidth);
    }

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("\n程序完成!\n");

    return 0;
}
