/**
 * Lesson 02: 矩阵乘法 (参考答案)
 *
 * 编译：nvcc -o matrix_mul_solution matrix_mul.cu
 * 运行：./matrix_mul_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 512  // 矩阵大小 (N x N)
#define BLOCK_SIZE 16  // 块大小

// ============================================================================
// 基础版本：直接全局内存访问
// ============================================================================

__global__ void matrix_mul_basic(float* A, float* B, float* C, int n) {
    // 计算二维线程 ID
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引

    // 边界检查
    if (col < n && row < n) {
        float sum = 0.0f;
        // 计算点积：C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ============================================================================
// 优化版本：使用共享内存
// ============================================================================

__global__ void matrix_mul_shared(float* A, float* B, float* C, int n) {
    // 声明共享内存 - 每个 block 共享
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 计算线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算线程对应的输出矩阵位置
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 分块计算
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 加载数据到共享内存
        if (row < n && tile * BLOCK_SIZE + tx < n) {
            As[ty][tx] = A[row * n + tile * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (tile * BLOCK_SIZE + ty < n && col < n) {
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 确保所有线程都完成加载
        __syncthreads();

        // 从共享内存计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 确保所有线程都完成计算
        __syncthreads();
    }

    // 写入结果
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

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

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 02: 矩阵乘法 ===\n\n");
    printDeviceinfo();

    int size = N * N * sizeof(float);

    // 分配 host 内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);

    // 初始化矩阵
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

    // 运行并计时各个版本
    float time_basic, time_shared, time_cpu;

    // 基础版本
    {
        SimpleTimer timer;
        timer.startRecord();
        matrix_mul_basic<<<grid, threads>>>(d_A, d_B, d_C, N);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        time_basic = timer.elapsedMs();
        printf("基础版本 GPU 时间：%.3f ms\n", time_basic);
    }

    // 共享内存优化版本
    {
        SimpleTimer timer;
        timer.startRecord();
        matrix_mul_shared<<<grid, threads>>>(d_A, d_B, d_C, N);
        timer.stopRecord();
        CUDA_CHECK_KERNEL();
        time_shared = timer.elapsedMs();
        printf("共享内存优化：%.3f ms (加速 %.2fx)\n",
               time_shared, time_basic / time_shared);
    }

    // 复制结果回 host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // CPU 版本
    {
        SimpleTimer timer;
        timer.startRecord();
        matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
        timer.stopRecord();
        time_cpu = timer.elapsedMs();
        printf("CPU 参考时间：%.3f ms (GPU 加速 %.1fx)\n\n",
               time_cpu, time_cpu / time_basic);
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
