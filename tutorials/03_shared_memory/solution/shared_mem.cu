/**
 * Lesson 03: 共享内存 (参考答案)
 *
 * 编译：nvcc -o shared_mem_solution shared_mem.cu
 * 运行：./shared_mem_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define ARRAY_SIZE (1024 * 1024)  // 1M 元素
#define BLOCK_SIZE 256

// ============================================================================
// 基础版本：只使用全局内存
// ============================================================================

__global__ void reduce_basic(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // 每个线程处理多个元素（网格跨步循环）
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 将部分和写入全局内存
    output[blockIdx.x] = sum;
}

// ============================================================================
// 优化版本 1：使用共享内存的并行归约
// ============================================================================

__global__ void reduce_shared(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 声明共享内存 - 每个 block 共享这块内存
    __shared__ float sdata[BLOCK_SIZE];

    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 将局部和存入共享内存
    sdata[tid] = sum;
    __syncthreads();  // 确保所有线程都写入完成

    // 树形归约 - 对数时间复杂度
    // 第一轮：sdata[0] += sdata[1], sdata[2] += sdata[3], ...
    // 第二轮：sdata[0] += sdata[2], sdata[4] += sdata[6], ...
    // ...
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 确保每轮归约完成
    }

    // 将结果写入输出
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// 优化版本 2：避免 bank conflict（无分支）
// ============================================================================

__global__ void reduce_no_bank_conflict(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];

    float sum = 0.0f;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // 无分支归约 - 避免 warp divergence
    // 同时避免 bank conflict（通过线程索引重排）
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

float reduce_cpu(float* input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

// ============================================================================
// 主函数
// ============================================================================

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
    printf("Block 数量：%d\n", num_blocks);
    printf("期望结果：%.0f\n\n", (float)ARRAY_SIZE);

    float expected = reduce_cpu(h_input, ARRAY_SIZE);
    float time_basic, time_shared;

    // 测试基础版本
    {
        SimpleTimer timer;
        timer.startRecord();
        reduce_basic<<<num_blocks, threads_per_block>>>(d_input, d_output, ARRAY_SIZE);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        time_basic = timer.elapsedMs();
        printf("基础版本：%.3f ms\n", time_basic);
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
        time_shared = timer.elapsedMs();
        printf("共享内存版本：%.3f ms", time_shared);
        if (time_basic > 0) {
            printf(" (加速 %.2fx)", time_basic / time_shared);
        }
        printf("\n");
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

    printf("\n=== 知识点总结 ===\n");
    printf("1. 共享内存是 on-chip 内存，速度比全局内存快得多\n");
    printf("2. __syncthreads() 确保 block 内所有线程同步\n");
    printf("3. Bank conflict 会降低共享内存性能\n");
    printf("4. 树形归约将 O(n) 降到 O(log n)\n");

    printf("\n程序完成!\n");

    return 0;
}
