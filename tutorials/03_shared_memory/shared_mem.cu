/**
 * Lesson 03: 共享内存
 *
 * 目标：深入学习共享内存和 bank conflict
 *
 * TODO:
 * 1. 理解共享内存的工作原理
 * 2. 实现使用共享内存的归约求和
 * 3. 观察 bank conflict 的影响
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define ARRAY_SIZE (1024 * 1024)  // 1M 元素
#define BLOCK_SIZE 256

// 基础归约：只使用全局内存
__global__ void reduce_basic(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 将部分和写入全局内存
    output[blockIdx.x] = sum;
}

// 使用共享内存的归约
// TODO: 实现 block 内的并行归约
__global__ void reduce_shared(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // TODO: 声明共享内存
    // __shared__ float sdata[BLOCK_SIZE];

    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // TODO: 将局部和存入共享内存
    // sdata[tid] = sum;
    // __syncthreads();

    // TODO: 实现树形归约
    // for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // TODO: 将结果写入输出
    // if (tid == 0) {
    //     output[blockIdx.x] = sdata[0];
    // }
}

// CPU 版本用于验证
float reduce_cpu(float* input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

int main() {
    printf("=== Lesson 03: 共享内存 ===\n\n");
    printDeviceinfo();

    int size = ARRAY_SIZE * sizeof(float);

    // 分配 host 内存
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(BLOCK_SIZE * sizeof(float));

    // 初始化数据
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = 1.0f;  // 全 1 数组，和应该等于元素个数
    }

    // 分配 device 内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, BLOCK_SIZE * sizeof(float)));

    // 复制数据
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int num_blocks = BLOCK_SIZE;
    int threads_per_block = BLOCK_SIZE;

    printf("数组大小：%d\n", ARRAY_SIZE);
    printf("期望结果：%.0f\n\n", (float)ARRAY_SIZE);

    float expected = reduce_cpu(h_input, ARRAY_SIZE);

    // 测试基础版本
    {
        SimpleTimer timer;
        timer.startRecord();
        reduce_basic<<<num_blocks, threads_per_block>>>(d_input, d_output, ARRAY_SIZE);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("基础版本：%.3f ms\n", timer.elapsedMs());
    }

    // 复制部分结果回 host 并求和
    CUDA_CHECK(cudaMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_basic = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) sum_basic += h_output[i];
    printf("  结果：%.0f (误差：%.0f)\n\n", sum_basic, sum_basic - expected);

    // 测试共享内存版本
    {
        SimpleTimer timer;
        timer.startRecord();
        reduce_shared<<<num_blocks, threads_per_block>>>(d_input, d_output, ARRAY_SIZE);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("共享内存版本：%.3f ms\n", timer.elapsedMs());
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_output, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_shared = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) sum_shared += h_output[i];
    printf("  结果：%.0f (误差：%.0f)\n", sum_shared, sum_shared - expected);

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("\n程序完成!\n");

    return 0;
}
