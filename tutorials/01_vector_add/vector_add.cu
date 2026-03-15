/**
 * Lesson 01: 向量加法
 *
 * 目标：学习 CUDA 内存管理和并行计算
 *
 * TODO:
 * 1. 分配 device 内存 (cudaMalloc)
 * 2. 将数据从 host 复制到 device (cudaMemcpy)
 * 3. 实现向量加法 kernel
 * 4. 将结果从 device 复制回 host
 * 5. 释放 device 内存 (cudaFree)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 1000000  // 向量大小

// TODO: 实现向量加法 kernel
// 计算 C[i] = A[i] + B[i]
__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    // 计算全局线程 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: 确保不越界，然后执行加法
    // if (idx < n) {
    //     c[idx] = a[idx] + b[idx];
    // }
}

int main() {
    printf("=== Lesson 01: 向量加法 ===\n\n");
    printDeviceinfo();

    // 分配 host 内存
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // TODO: 声明 device 指针
    float *d_a, *d_b, *d_c;

    // TODO: 1. 分配 device 内存
    // cudaMalloc(&d_a, N * sizeof(float));
    // cudaMalloc(&d_b, N * sizeof(float));
    // cudaMalloc(&d_c, N * sizeof(float));

    // TODO: 2. 将数据从 host 复制到 device
    // cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 kernel 参数
    int threads_per_block = 256;
    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("向量大小：%d\n", N);
    printf("启动 kernel: %d blocks x %d threads\n\n", number_of_blocks, threads_per_block);

    // TODO: 3. 启动 kernel
    // vector_add_kernel<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // TODO: 4. 将结果从 device 复制回 host
    // cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: 5. 释放 device 内存
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    // 验证结果（前 10 个元素）
    printf("结果验证（前 10 个元素）:\n");
    for (int i = 0; i < 10; i++) {
        printf("  h_c[%d] = %.1f (期望：%.1f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // 清理 host 内存
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n程序完成!\n");

    return 0;
}
