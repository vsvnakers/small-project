/**
 * Lesson 04: 原子操作 (参考答案)
 *
 * 编译：nvcc -o atomic_ops_solution atomic_ops.cu
 * 运行：./atomic_ops_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define ARRAY_SIZE 1000000
#define NUM_BUCKETS 256

// ============================================================================
// 原子计数器
// ============================================================================

__global__ void atomic_counter_basic(int* counter, int n) {
    // atomicAdd 是原子操作，确保多个线程同时访问时不会丢失更新
    atomicAdd(counter, 1);
}

// ============================================================================
// 原子直方图
// ============================================================================

__global__ void atomic_histogram(unsigned char* input, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        unsigned char value = input[idx];
        // 原子地增加对应 bucket 的计数
        atomicAdd(&histogram[value], 1);
    }
}

// ============================================================================
// 优化的直方图（使用共享内存）
// ============================================================================

__global__ void atomic_histogram_shared(unsigned char* input, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每个 block 的共享内存直方图
    __shared__ int local_hist[NUM_BUCKETS];

    // 初始化共享内存
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // 在共享内存中累积
    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&local_hist[value], 1);
    }
    __syncthreads();

    // 合并到全局内存
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        if (local_hist[i] > 0) {
            atomicAdd(&histogram[i], local_hist[i]);
        }
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void histogram_cpu(unsigned char* input, int* histogram, int n, int buckets) {
    for (int i = 0; i < buckets; i++) histogram[i] = 0;
    for (int i = 0; i < n; i++) {
        histogram[input[i]]++;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 04: 原子操作 ===\n\n");
    printDeviceinfo();

    // -------------------------------------------------------------------------
    // 测试 1: 原子计数器
    // -------------------------------------------------------------------------
    printf("测试 1: 原子计数器\n");
    printf("------------------\n");

    int *d_counter, h_counter = 0;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    int threads = 10000;
    int blocks = 100;
    int expected_count = threads * blocks;

    printf("启动 %d x %d = %d 线程\n", blocks, threads, expected_count);

    SimpleTimer timer;
    timer.startRecord();
    atomic_counter_basic<<<blocks, threads>>>(d_counter, expected_count);
    timer.stopRecord();
    CUDA_CHECK_KERNEL();
    printf("GPU 时间：%.3f ms\n", timer.elapsedMs());

    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("计数器结果：%d (期望：%d) %s\n\n", h_counter, expected_count,
           (h_counter == expected_count) ? "✓" : "✗");

    CUDA_CHECK(cudaFree(d_counter));

    // -------------------------------------------------------------------------
    // 测试 2: 原子直方图
    // -------------------------------------------------------------------------
    printf("测试 2: 原子直方图\n");
    printf("------------------\n");

    unsigned char *h_input = (unsigned char*)malloc(ARRAY_SIZE * sizeof(unsigned char));
    int *h_histogram = (int*)malloc(NUM_BUCKETS * sizeof(int));
    int *h_histogram_gpu = (int*)malloc(NUM_BUCKETS * sizeof(int));

    // 生成随机数据
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (i * 7) % NUM_BUCKETS;  // 伪随机分布
    }

    int *d_histogram;
    unsigned char *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int num_blocks = (ARRAY_SIZE + threads_per_block - 1) / threads_per_block;

    // GPU 版本
    timer.startRecord();
    atomic_histogram<<<num_blocks, threads_per_block>>>(d_input, d_histogram, ARRAY_SIZE);
    timer.stopRecord();
    CUDA_CHECK_KERNEL();
    printf("GPU 时间：%.3f ms\n", timer.elapsedMs());

    CUDA_CHECK(cudaMemcpy(h_histogram_gpu, d_histogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost));

    // CPU 版本
    timer.startRecord();
    histogram_cpu(h_input, h_histogram, ARRAY_SIZE, NUM_BUCKETS);
    printf("CPU 时间：%.3f ms\n\n", timer.elapsedMs());

    // 验证结果
    printf("结果验证 (前 10 个 bucket):\n");
    bool all_correct = true;
    for (int i = 0; i < 10; i++) {
        bool match = (h_histogram_gpu[i] == h_histogram[i]);
        if (!match) all_correct = false;
        printf("  Bucket %d: GPU=%d, CPU=%d %s\n",
               i, h_histogram_gpu[i], h_histogram[i],
               match ? "✓" : "✗");
    }
    printf("  ... (共 %d 个 buckets)\n", NUM_BUCKETS);

    // 完整验证
    for (int i = 10; i < NUM_BUCKETS; i++) {
        if (h_histogram_gpu[i] != h_histogram[i]) {
            all_correct = false;
            break;
        }
    }

    printf("\n验证结果：%s\n", all_correct ? "全部正确 ✓" : "有错误 ✗");

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));
    free(h_input);
    free(h_histogram);
    free(h_histogram_gpu);

    printf("\n=== 知识点总结 ===\n");
    printf("1. 原子操作确保多线程安全访问同一内存位置\n");
    printf("2. CUDA 提供 atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax 等\n");
    printf("3. 原子操作会降低性能，必要时才使用\n");
    printf("4. 可用共享内存减少全局原子操作次数\n");

    printf("\n程序完成!\n");

    return 0;
}
