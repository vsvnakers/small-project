/**
 * Lesson 04: 原子操作
 *
 * 目标：学习原子操作和线程同步
 *
 * TODO:
 * 1. 理解原子操作的必要性
 * 2. 实现原子计数器
 * 3. 实现原子直方图
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define ARRAY_SIZE 1000000
#define NUM_BUCKETS 256

// TODO: 实现原子计数器
// 每个线程都对同一个计数器加 1
__global__ void atomic_counter_basic(int* counter, int n) {
    // TODO: 使用 atomicAdd 实现
    // atomicAdd(counter, 1);
}

// TODO: 实现原子直方图
// 统计输入数据中每个值出现的次数
__global__ void atomic_histogram(unsigned char* input, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 边界检查
    // if (idx < n) {
    //     unsigned char value = input[idx];
    //     atomicAdd(&histogram[value], 1);
    // }
}

// CPU 版本用于验证
void histogram_cpu(unsigned char* input, int* histogram, int n, int buckets) {
    for (int i = 0; i < buckets; i++) histogram[i] = 0;
    for (int i = 0; i < n; i++) {
        histogram[input[i]]++;
    }
}

int main() {
    printf("=== Lesson 04: 原子操作 ===\n\n");
    printDeviceinfo();

    // 测试 1: 原子计数器
    printf("测试 1: 原子计数器\n");
    printf("------------------\n");

    int *d_counter, h_counter = 0;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    int threads = 10000;
    int blocks = 100;

    atomic_counter_basic<<<blocks, threads>>>(d_counter, threads * blocks);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("线程总数：%d\n", threads * blocks);
    printf("计数器结果：%d (期望：%d)\n\n", h_counter, threads * blocks);

    CUDA_CHECK(cudaFree(d_counter));

    // 测试 2: 原子直方图
    printf("测试 2: 原子直方图\n");
    printf("------------------\n");

    unsigned char *h_input = (unsigned char*)malloc(ARRAY_SIZE * sizeof(unsigned char));
    int *h_histogram = (int*)malloc(NUM_BUCKETS * sizeof(int));
    int *h_histogram_gpu = (int*)malloc(NUM_BUCKETS * sizeof(int));

    // 生成随机数据
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = i % NUM_BUCKETS;
    }

    int *d_histogram;
    unsigned char *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int num_blocks = (ARRAY_SIZE + threads_per_block - 1) / threads_per_block;

    {
        SimpleTimer timer;
        timer.startRecord();
        atomic_histogram<<<num_blocks, threads_per_block>>>(d_input, d_histogram, ARRAY_SIZE);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("GPU 时间：%.3f ms\n", timer.elapsedMs());
    }

    CUDA_CHECK(cudaMemcpy(h_histogram_gpu, d_histogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

    // CPU 版本
    SimpleTimer timer;
    timer.startRecord();
    histogram_cpu(h_input, h_histogram, ARRAY_SIZE, NUM_BUCKETS);
    printf("CPU 时间：%.3f ms\n\n", timer.elapsedMs());

    // 验证结果
    printf("结果验证 (前 10 个 bucket):\n");
    for (int i = 0; i < 10; i++) {
        printf("  Bucket %d: GPU=%d, CPU=%d %s\n",
               i, h_histogram_gpu[i], h_histogram[i],
               (h_histogram_gpu[i] == h_histogram[i]) ? "✓" : "✗");
    }

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));
    free(h_input);
    free(h_histogram);
    free(h_histogram_gpu);

    printf("\n程序完成!\n");

    return 0;
}
