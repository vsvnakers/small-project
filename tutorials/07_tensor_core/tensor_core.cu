/**
 * Lesson 07: Tensor Core
 *
 * 目标：学习 Tensor Core 编程，使用 WMMA API
 *
 * 前提：需要 Volta (V100) 或更新的 GPU，Ampere (RTX 30) / Ada (RTX 40)
 *
 * TODO:
 * 1. 理解 Tensor Core 的工作原理
 * 2. 使用 WMMA API 进行矩阵乘法
 * 3. 比较 Tensor Core 和 CUDA Core 的性能
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include "common.h"

// Tensor Core 要求的矩阵大小
#define M 16
#define N 16
#define K 16

using namespace nvcuda;

// TODO: 使用 WMMA API 实现矩阵乘法
// C = A * B, 其中 A, B, C 都是 16x16 矩阵
__global__ void tensor_core_mma(float* A, float* B, float* C, int n) {
    // WMMA 类型定义
    // wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b_frag;
    // wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // TODO:
    // 1. 声明 WMMA fragment
    // 2. 从全局内存加载数据到 fragment
    // 3. 执行 wmma::mma_sync
    // 4. 将结果存回全局内存
}

// 标准的 CUDA Core 矩阵乘法（用于对比）
__global__ void cuda_core_gemm(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    printf("=== Lesson 07: Tensor Core ===\n\n");
    printDeviceinfo();

    // 检查 Tensor Core 支持
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 7) {
        printf("警告：您的 GPU 不支持 Tensor Core (需要计算能力 7.0+)\n");
        printf("本示例将跳过 Tensor Core 测试\n\n");
    } else {
        printf("Tensor Core 支持：是 (SM %d.%d)\n\n", prop.major, prop.minor);
    }

    // 矩阵大小（必须是 16 的倍数）
    int matrix_size = 1024;  // 1024 x 1024 矩阵
    int size = matrix_size * matrix_size * sizeof(float);

    // 分配内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_tensor = (float*)malloc(size);
    float *h_C_cuda = (float*)malloc(size);

    // 初始化矩阵
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)(i % 50) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 配置 kernel
    dim3 threads(16, 16);
    dim3 grid((matrix_size + 15) / 16, (matrix_size + 15) / 16);

    printf("矩阵大小：%d x %d\n", matrix_size, matrix_size);
    printf("启动配置：%dx%d threads, %dx%d grid\n\n",
           threads.x, threads.y, grid.x, grid.y);

    // 测试 CUDA Core 版本
    printf("测试 1: CUDA Core GEMM\n");
    printf("----------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();
        cuda_core_gemm<<<grid, threads>>>(d_A, d_B, d_C, matrix_size);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("时间：%.3f ms\n", timer.elapsedMs());
    }

    CUDA_CHECK(cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost));

    // 测试 Tensor Core 版本
    if (prop.major >= 7) {
        printf("\n测试 2: Tensor Core WMMA\n");
        printf("----------------------\n");
        {
            SimpleTimer timer;
            timer.startRecord();

            // TODO: 调用 tensor_core_mma kernel

            timer.stopRecord();
            CUDA_CHECK_KERNEL();
            printf("时间：%.3f ms (待实现)\n", timer.elapsedMs());
        }
    }

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_tensor);
    free(h_C_cuda);

    printf("\n程序完成!\n");

    return 0;
}
