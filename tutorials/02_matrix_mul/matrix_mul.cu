/**
 * Lesson 02: 矩阵乘法
 *
 * 目标：学习二维线程块和共享内存优化
 *
 * TODO:
 * 1. 实现基础的矩阵乘法 kernel（不使用共享内存）
 * 2. 理解二维索引计算
 * 3. 实现使用共享内存的优化版本
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 512  // 矩阵大小 (N x N)
#define BLOCK_SIZE 16  // 块大小

// 基础版本：直接全局内存访问
// TODO: 实现矩阵乘法 C = A * B
// C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to N-1
__global__ void matrix_mul_basic(float* A, float* B, float* C, int n) {
    // 计算二维线程 ID
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引

    // TODO: 边界检查并实现矩阵乘法
    // if (col < n && row < n) {
    //     float sum = 0.0f;
    //     for (int k = 0; k < n; k++) {
    //         sum += A[row * n + k] * B[k * n + col];
    //     }
    //     C[row * n + col] = sum;
    // }
}

// 优化版本：使用共享内存
__global__ void matrix_mul_shared(float* A, float* B, float* C, int n) {
    // TODO: 实现使用共享内存的矩阵乘法
    // 提示：
    // 1. 声明共享内存 __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    // 2. 分块加载数据到共享内存
    // 3. 同步线程 __syncthreads()
    // 4. 从共享内存计算
}

// CPU 版本的矩阵乘法（用于验证）
void matrix_mul_cpu(float* A, float* B, float* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    printf("=== Lesson 02: 矩阵乘法 ===\n\n");
    printDeviceinfo();

    int size = N * N * sizeof(float);

    // 分配 host 内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);

    // 初始化矩阵（使用小值避免溢出）
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 10) / 10.0f;
        h_B[i] = (float)(i % 7) / 10.0f;
    }

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 复制数据到 device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 配置二维线程块
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("矩阵大小：%d x %d\n", N, N);
    printf("线程块：%dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Grid: %dx%d\n\n", grid.x, grid.y);

    // 运行基础版本
    {
        SimpleTimer timer;
        timer.startRecord();
        matrix_mul_basic<<<grid, threads>>>(d_A, d_B, d_C, N);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        printf("基础版本 GPU 时间：%.3f ms\n", timer.elapsedMs());
    }

    // 复制结果回 host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // CPU 版本（用于验证）
    {
        SimpleTimer timer;
        timer.startRecord();
        matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
        timer.stopRecord();
        printf("CPU 参考时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 验证结果
    verifyResults(h_C_cpu, h_C_gpu, N * N, "矩阵乘法");

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    printf("\n程序完成!\n");

    return 0;
}
