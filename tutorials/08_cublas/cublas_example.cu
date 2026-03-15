/**
 * Lesson 08: cuBLAS 库
 *
 * 目标：学习使用 NVIDIA cuBLAS 库进行高性能线性代数运算
 *
 * TODO:
 * 1. 了解 cuBLAS API 基础
 * 2. 使用 cuBLAS 进行 GEMM (矩阵乘法)
 * 3. 使用 cuBLAS 进行批量矩阵运算
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "common.h"

#define M 1024
#define K 1024
#define N 1024

// TODO: 使用 cuBLAS 实现矩阵乘法 C = A * B
void cublas_gemm(cublasHandle_t handle,
                 float* d_A, float* d_B, float* d_C,
                 int m, int n, int k) {
    // cuBLAS 使用列优先存储
    // C = alpha * A * B + beta * C
    //
    // TODO:
    // 1. 设置 alpha = 1.0, beta = 0.0
    // 2. 调用 cublasSgemm

    // cublasStatus_t status = cublasSgemm(
    //     handle,
    //     CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
    //     m, n, k,
    //     &alpha,
    //     d_A, m,      // A 是 m x k
    //     d_B, k,      // B 是 k x n
    //     &beta,
    //     d_C, m       // C 是 m x n
    // );
}

// CPU 版本用于验证
void gemm_cpu(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    printf("=== Lesson 08: cuBLAS 库 ===\n\n");
    printDeviceinfo();

    // 创建 cuBLAS handle
    cublasHandle_t cublas_handle;
    // TODO: 初始化 cublas_handle
    // cublasCreate(&cublas_handle);

    // 分配内存
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_a);
    float *h_B = (float*)malloc(size_b);
    float *h_C = (float*)malloc(size_c);
    float *h_C_ref = (float*)malloc(size_c);

    // 初始化矩阵
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(i % 50) / 100.0f;
    }

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_a));
    CUDA_CHECK(cudaMalloc(&d_B, size_b));
    CUDA_CHECK(cudaMalloc(&d_C, size_c));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    printf("矩阵 A: %d x %d\n", M, K);
    printf("矩阵 B: %d x %d\n", K, N);
    printf("矩阵 C: %d x %d\n\n", M, N);

    // 测试 cuBLAS
    printf("测试：cuBLAS GEMM\n");
    printf("---------------\n");

    // TODO: 调用 cublas_gemm

    // TODO: 同步并获取结果
    // CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));

    printf("结果：待实现\n");

    // CPU 参考（使用小矩阵验证）
    #if 0
    printf("\n验证 (10x10 子矩阵):\n");
    for (int i = 0; i < 10; i++) {
        printf("  ");
        for (int j = 0; j < 10; j++) {
            printf("%6.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    #endif

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // TODO: 销毁 cuBLAS handle
    // cublasDestroy(cublas_handle);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    printf("\n程序完成!\n");

    return 0;
}
