/**
 * Lesson 01: 向量加法 (参考答案)
 *
 * 编译：nvcc -o vector_add_solution vector_add.cu
 * 运行：./vector_add_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 1000000  // 向量大小

// 向量加法 kernel
__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    // 计算全局线程 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，确保不越界
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
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

    // 声明 device 指针
    float *d_a, *d_b, *d_c;

    // 1. 分配 device 内存
    printf("分配 device 内存：%.2f MB\n", (3 * N * sizeof(float)) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 2. 将数据从 host 复制到 device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // 配置 kernel 参数
    int threads_per_block = 256;
    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("向量大小：%d\n", N);
    printf("启动 kernel: %d blocks x %d threads\n", number_of_blocks, threads_per_block);

    // 创建计时器
    SimpleTimer timer;

    // 3. 启动 kernel
    timer.startRecord();
    vector_add_kernel<<<number_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    timer.stopRecord();

    // 检查 kernel 启动错误
    CUDA_CHECK_KERNEL();

    printf("Kernel 执行时间：%.3f ms\n\n", timer.elapsedMs());

    // 4. 将结果从 device 复制回 host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果（前 10 个元素）
    printf("结果验证（前 10 个元素）:\n");
    for (int i = 0; i < 10; i++) {
        printf("  h_c[%d] = %.1f (期望：%.1f) %s\n",
               i, h_c[i], h_a[i] + h_b[i],
               (h_c[i] == h_a[i] + h_b[i]) ? "✓" : "✗");
    }
    printf("  ...\n");

    // 5. 释放 device 内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 清理 host 内存
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n程序完成!\n");

    return 0;
}
