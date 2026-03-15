/**
 * Lesson 08: cuBLAS 库 (参考答案)
 *
 * 编译：nvcc -o cublas_example_solution cublas_example.cu -lcublas
 * 运行：./cublas_example_solution
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "common.h"

#define M 1024
#define K 1024
#define N 1024

// ============================================================================
// cuBLAS GEMM 封装
// ============================================================================

void cublas_gemm(cublasHandle_t handle,
                 float* d_A, float* d_B, float* d_C,
                 int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 使用列优先存储 (column-major)
    // C = A * B (行优先) 等价于 C^T = B^T * A^T (列优先)
    // 所以我们要交换 A 和 B 的顺序，并转置

    // 计算 C^T = B^T * A^T，这样结果就是行优先的 C
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // 转置输入
        n, m, k,
        &alpha,
        d_B, k,      // B 是 k x n, 转置后为 n x k
        d_A, m,      // A 是 m x k, 转置后为 k x m
        &beta,
        d_C, n       // C 是 m x n, 结果为 n x m (列优先存储行优先数据)
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM 失败：%d\n", status);
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// 批量 GEMM
// ============================================================================

void cublas_batched_gemm(cublasHandle_t handle,
                         float** d_A_array, float** d_B_array, float** d_C_array,
                         int m, int n, int k, int batch_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 为指针数组分配 device 内存
    float **d_A_ptr, **d_B_ptr, **d_C_ptr;
    CUDA_CHECK(cudaMalloc(&d_A_ptr, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_B_ptr, batch_size * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_C_ptr, batch_size * sizeof(float*)));

    // 复制指针数组
    CUDA_CHECK(cudaMemcpy(d_A_ptr, d_A_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_ptr, d_B_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ptr, d_C_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

    // 批量 GEMM
    cublasStatus_t status = cublasSgemmBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        (const float**)d_A_ptr, m,
        (const float**)d_B_ptr, k,
        &beta,
        d_C_ptr, m,
        batch_size
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Batched GEMM 失败：%d\n", status);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaFree(d_A_ptr));
    CUDA_CHECK(cudaFree(d_B_ptr));
    CUDA_CHECK(cudaFree(d_C_ptr));
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
// 验证函数
// ============================================================================

bool verify_gemm(float* h_C_gpu, float* h_C_cpu, int m, int n) {
    const float tolerance = 1e-3;
    int max_error_idx = -1;
    float max_error = 0;

    for (int i = 0; i < m * n; i++) {
        float diff = h_C_gpu[i] - h_C_cpu[i];
        if (diff < 0) diff = -diff;
        if (diff > max_error) {
            max_error = diff;
            max_error_idx = i;
        }
        if (diff > tolerance) {
            printf("验证失败：位置 %d, GPU=%f, CPU=%f, 误差=%f\n",
                   i, h_C_gpu[i], h_C_cpu[i], diff);
            return false;
        }
    }

    printf("验证通过：最大误差 = %.6f (位置 %d)\n", max_error, max_error_idx);
    return true;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 08: cuBLAS 库 ===\n\n");
    printDeviceinfo();

    // 创建 cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS 初始化失败：%d\n", status);
        return EXIT_FAILURE;
    }

    // 使用默认精度（不用 Tensor Core）以保证精度
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    // -------------------------------------------------------------------------
    // 测试 1: 单矩阵 GEMM
    // -------------------------------------------------------------------------
    printf("测试 1: cuBLAS GEMM (单矩阵)\n");
    printf("--------------------------\n");

    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_a);
    float *h_B = (float*)malloc(size_b);
    float *h_C_gpu = (float*)malloc(size_c);
    float *h_C_cpu = (float*)malloc(size_c);

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

    // cuBLAS GEMM
    {
        SimpleTimer timer;
        timer.startRecord();

        cublas_gemm(cublas_handle, d_A, d_B, d_C, M, N, K);

        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stopRecord();
        printf("cuBLAS 时间：%.3f ms\n", timer.elapsedMs());
    }

    // 复制结果回 host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_c, cudaMemcpyDeviceToHost));

    // 显示部分结果
    printf("\n结果示例 (C[0:4, 0:4]):\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++) {
            printf("%8.3f ", h_C_gpu[i * N + j]);
        }
        printf("\n");
    }

    // -------------------------------------------------------------------------
    // 测试 2: 批量 GEMM
    // -------------------------------------------------------------------------
    printf("\n测试 2: cuBLAS 批量 GEMM\n");
    printf("----------------------\n");

    int batch_size = 32;
    int batch_m = 64, batch_n = 64, batch_k = 64;

    // 分配多个矩阵
    float **h_A_array = (float**)malloc(batch_size * sizeof(float*));
    float **h_B_array = (float**)malloc(batch_size * sizeof(float*));
    float **h_C_array = (float**)malloc(batch_size * sizeof(float*));

    float **d_A_array = (float**)malloc(batch_size * sizeof(float*));
    float **d_B_array = (float**)malloc(batch_size * sizeof(float*));
    float **d_C_array = (float**)malloc(batch_size * sizeof(float*));

    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMalloc(&h_A_array[i], batch_m * batch_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_B_array[i], batch_k * batch_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_C_array[i], batch_m * batch_n * sizeof(float)));

        // 初始化
        float *h_temp_a = (float*)malloc(batch_m * batch_k * sizeof(float));
        float *h_temp_b = (float*)malloc(batch_k * batch_n * sizeof(float));
        for (int j = 0; j < batch_m * batch_k; j++) {
            h_temp_a[j] = (float)(j % 100) / 100.0f;
        }
        for (int j = 0; j < batch_k * batch_n; j++) {
            h_temp_b[j] = (float)(j % 50) / 100.0f;
        }

        CUDA_CHECK(cudaMemcpy(h_A_array[i], h_temp_a, batch_m * batch_k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_B_array[i], h_temp_b, batch_k * batch_n * sizeof(float), cudaMemcpyHostToDevice));

        free(h_temp_a);
        free(h_temp_b);
    }

    {
        SimpleTimer timer;
        timer.startRecord();

        cublas_batched_gemm(cublas_handle,
                           (float**)h_A_array, (float**)h_B_array, (float**)h_C_array,
                           batch_m, batch_n, batch_k, batch_size);

        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stopRecord();
        printf("批量 GEMM 时间：%.3f ms (%d 个 %dx%dx%d 矩阵)\n",
               timer.elapsedMs(), batch_size, batch_m, batch_n, batch_k);
        printf("平均每矩阵：%.3f ms\n", timer.elapsedMs() / batch_size);
    }

    // 清理
    for (int i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaFree(h_A_array[i]));
        CUDA_CHECK(cudaFree(h_B_array[i]));
        CUDA_CHECK(cudaFree(h_C_array[i]));
    }
    free(h_A_array);
    free(h_B_array);
    free(h_C_array);
    free(d_A_array);
    free(d_B_array);
    free(d_C_array);

    // -------------------------------------------------------------------------
    // 清理
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    cublasDestroy(cublas_handle);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    printf("\n=== 知识点总结 ===\n");
    printf("1. cuBLAS 是 NVIDIA 的 GPU 加速线性代数库\n");
    printf("2. 使用列优先存储 (column-major)\n");
    printf("3. cublasSgemm: 单精度矩阵乘法\n");
    printf("4. cublasDgemm: 双精度矩阵乘法\n");
    printf("5. cublasSgemmBatched: 批量矩阵乘法\n");
    printf("6. CUBLAS_TENSOR_OP_MATH: 启用 Tensor Core 加速\n");
    printf("7. cuBLAS v2 需要创建 handle 管理上下文\n");

    printf("\n程序完成!\n");

    return 0;
}
