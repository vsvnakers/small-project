/**
 * Lesson 07: Tensor Core (参考答案 - 简化版)
 *
 * 编译：nvcc -o tensor_core_solution tensor_core.cu -arch=sm_86
 * 运行：./tensor_core_solution
 *
 * 注意：需要 Volta (V100) 或更新的 GPU (计算能力 7.0+)
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include "common.h"

using namespace nvcuda;

// ============================================================================
// WMMA 配置
// ============================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Tensor Core 示例 - 使用 WMMA API 进行矩阵乘法
// ============================================================================

__global__ void tensor_core_wmma(const half* A, const half* B, float* C, int n) {
    // 每个 warp 负责一个 16x16 的输出块
    int warpId = (threadIdx.x / 32) + (blockIdx.x * blockDim.y + threadIdx.y) * (blockDim.x / 32);

    int block_row = warpId / ((n + WMMA_N - 1) / WMMA_N);
    int block_col = warpId % ((n + WMMA_N - 1) / WMMA_N);

    int row_start = block_row * WMMA_M;
    int col_start = block_col * WMMA_N;

    // 声明 WMMA fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化 accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // 边界检查
    if (row_start >= n || col_start >= n) return;

    // 直接加载数据并执行 MMA
    // 简化版本：假设矩阵大小是 16 的倍数
    for (int k = 0; k < WMMA_K; k += WMMA_K) {
        // 加载 A 和 B
        for (int t = 0; t < WMMA_M * WMMA_K / 32; t++) {
            int lane = threadIdx.x % 32;
            int idx = t * 32 + lane;
            if (idx < WMMA_M * WMMA_K) {
                int r = idx / WMMA_K;
                int c = idx % WMMA_K;
                if (row_start + r < n && k + c < n) {
                    a_frag.x[t] = A[(row_start + r) * n + (k + c)];
                }
            }
        }

        for (int t = 0; t < WMMA_K * WMMA_N / 32; t++) {
            int lane = threadIdx.x % 32;
            int idx = t * 32 + lane;
            if (idx < WMMA_K * WMMA_N) {
                int r = idx / WMMA_N;
                int c = idx % WMMA_N;
                if (k + r < n && col_start + c < n) {
                    b_frag.x[t] = B[(k + r) * n + (col_start + c)];
                }
            }
        }

        // 执行矩阵乘法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    for (int t = 0; t < WMMA_M * WMMA_N / 32; t++) {
        int lane = threadIdx.x % 32;
        int idx = t * 32 + lane;
        if (idx < WMMA_M * WMMA_N) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;
            if (row_start + r < n && col_start + c < n) {
                C[(row_start + r) * n + (col_start + c)] = c_frag.x[t];
            }
        }
    }
}

// ============================================================================
// 简单的 CUDA Core 版本用于对比
// ============================================================================

__global__ void cuda_core_gemm(const float* A, const float* B, float* C, int n) {
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

// ============================================================================
// CPU 参考实现
// ============================================================================

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

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 07: Tensor Core ===\n\n");
    printDeviceinfo();

    // 检查 Tensor Core 支持
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    bool has_tensor_core = (prop.major >= 7);

    if (!has_tensor_core) {
        printf("警告：您的 GPU 不支持 Tensor Core (需要计算能力 7.0+)\n");
        printf("本示例将跳过 Tensor Core 测试\n\n");
    } else {
        printf("Tensor Core 支持：是 (SM %d.%d)\n", prop.major, prop.minor);
        printf("Ampere 架构：支持 TF32, FP16, BF16, INT8\n\n");
    }

    // 矩阵大小（必须是 16 的倍数）
    int matrix_size = 256;
    int m = matrix_size, n = matrix_size, k = matrix_size;

    // 分配内存
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));
    float *h_C_ref = (float*)malloc(m * n * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < m * k; i++) {
        h_A[i] = (float)(i % 10) / 100.0f;
    }
    for (int i = 0; i < k * n; i++) {
        h_B[i] = (float)(i % 7) / 100.0f;
    }

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    printf("矩阵大小：%d x %d\n", m, n);
    printf("Grid/Block 配置：根据矩阵大小自动计算\n\n");

    if (has_tensor_core) {
        // CUDA Core 版本
        dim3 threads(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);

        {
            SimpleTimer timer;
            timer.startRecord();
            int iterations = 10;
            for (int i = 0; i < iterations; i++) {
                cuda_core_gemm<<<grid, threads>>>(d_A, d_B, d_C, matrix_size);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stopRecord();
            printf("CUDA Core 时间：%.3f ms (平均，%d 次迭代)\n",
                   timer.elapsedMs() / iterations, iterations);
        }

        CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // CPU 版本（用于验证）
    printf("\nCPU 参考实现 (256x256)...\n");
    {
        SimpleTimer timer;
        timer.startRecord();
        gemm_cpu(h_A, h_B, h_C_ref, m, n, k);
        timer.stopRecord();
        printf("CPU 时间：%.3f ms\n", timer.elapsedMs());
    }

    // 显示部分结果
    printf("\nCPU 结果 [0:4, 0:4]:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++) {
            printf("%8.5f ", h_C_ref[i * n + j]);
        }
        printf("\n");
    }

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    printf("\n=== 知识点总结 ===\n");
    printf("1. Tensor Core 是专用矩阵乘法单元\n");
    printf("2. WMMA API 需要 warp 级别协作\n");
    printf("3. 支持 FP16, BF16, TF32 (Ampere+), INT8 等精度\n");
    printf("4. 最小矩阵大小：16x16x16\n");
    printf("5. FP16 性能比 FP32 高 8-16 倍\n");
    printf("6. 注意精度损失：FP16 动态范围有限\n");

    printf("\n程序完成!\n");

    return 0;
}
