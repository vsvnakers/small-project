# Lesson 08: cuBLAS 库

## 学习目标

- 了解 cuBLAS 库的功能和 API
- 学习使用 cuBLAS 进行矩阵运算
- 掌握批量矩阵运算
- 理解性能优化技巧

## cuBLAS 简介

### 什么是 cuBLAS？

cuBLAS 是 NVIDIA 提供的**GPU 加速线性代数库**，实现了 BLAS (Basic Linear Algebra Subprograms) 标准。

### BLAS 级别

| 级别 | 操作 | 复杂度 |
|------|------|--------|
| Level 1 | 向量 - 向量 (点积、归约) | O(n) |
| Level 2 | 矩阵 - 向量 (GEMV) | O(n²) |
| Level 3 | 矩阵 - 矩阵 (GEMM) | O(n³) |

### cuBLAS 功能

- 矩阵乘法 (GEMM)
- 向量运算 (AXPY, DOT)
- 矩阵分解 (LU, QR, Cholesky)
- 特征值求解
- 奇异值分解 (SVD)

## 基本 API

### 初始化

```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// 使用...

cublasDestroy(handle);
```

### 设置数学模式

```cuda
// 启用 Tensor Core
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// 默认精度
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
```

## 矩阵乘法 (GEMM)

### API 说明

```cuda
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,    // A 的转置选项
    cublasOperation_t transb,    // B 的转置选项
    int m, int n, int k,         // 矩阵维度
    const float *alpha,          // 缩放因子 α
    const float *A, int lda,     // 矩阵 A 和 leading dimension
    const float *B, int ldb,     // 矩阵 B 和 leading dimension
    const float *beta,           // 缩放因子 β
    float *C, int ldc            // 矩阵 C 和 leading dimension
);
```

### 计算公式

```
C = α × A × B + β × C
```

### 简单示例

```cuda
#include <cublas_v2.h>

#define M 1024
#define K 1024
#define N 1024

void cublas_gemm() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 执行 GEMM (列优先)
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
                M, N, K,
                &alpha,
                d_A, M,      // A: M×K
                d_B, K,      // B: K×N
                &beta,
                d_C, M);     // C: M×N

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
}
```

### 列优先 vs 行优先

cuBLAS 使用**列优先**存储：

```
行优先 (C/C++):     列优先 (Fortran/cuBLAS):
[0,0] [0,1] [0,2]   [0,0] [1,0] [2,0]
[1,0] [1,1] [1,2]   [0,1] [1,1] [2,1]
[2,0] [2,1] [2,2]   [0,2] [1,2] [2,2]
```

### 行优先转换

对于行优先存储的矩阵，需要转置参数：

```
C = A × B (行优先)
等价于
C^T = B^T × A^T (列优先)
```

```cuda
// 行优先的 C = A × B
// 使用 cuBLAS 计算：
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,       // 注意维度交换
            &alpha,
            d_B, N,        // B 先 (转置后)
            d_A, K,        // A 后 (转置后)
            &beta,
            d_C, N);       // C (转置后)
```

## 批量 GEMM

### 为什么需要批量？

深度学习中的 batch 处理需要同时计算多个矩阵乘法。

### cublasSgemmBatched

```cuda
void batched_gemm(cublasHandle_t handle,
                  float** d_A_array, float** d_B_array, float** d_C_array,
                  int m, int n, int k, int batch_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 分配指针数组
    float **d_A_ptr, **d_B_ptr, **d_C_ptr;
    cudaMalloc(&d_A_ptr, batch_size * sizeof(float*));
    cudaMalloc(&d_B_ptr, batch_size * sizeof(float*));
    cudaMalloc(&d_C_ptr, batch_size * sizeof(float*));

    // 复制指针
    cudaMemcpy(d_A_ptr, d_A_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptr, d_B_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ptr, d_C_array, batch_size * sizeof(float*), cudaMemcpyHostToDevice);

    // 批量 GEMM
    cublasSgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k,
                       &alpha,
                       (const float**)d_A_ptr, m,
                       (const float**)d_B_ptr, k,
                       &beta,
                       d_C_ptr, m,
                       batch_size);

    cudaFree(d_A_ptr); cudaFree(d_B_ptr); cudaFree(d_C_ptr);
}
```

### stridedBatched GEMM

更高效的批量操作（内存连续）：

```cuda
cublasSgemmStridedBatched(handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          m, n, k,
                          &alpha,
                          d_A, m, stride_A,    // strideA = M×K
                          d_B, k, stride_B,    // strideB = K×N
                          &beta,
                          d_C, m, stride_C,    // strideC = M×N
                          batch_size);
```

## 性能优化

### 1. 矩阵维度对齐

```cuda
// 确保维度是 32 的倍数（Tensor Core 需要 16 的倍数）
int M_aligned = (M + 31) / 32 * 32;
```

### 2. 使用 Tensor Core

```cuda
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
```

### 3. 选择合适的数据类型

| 类型 | 性能 | 精度 | 推荐用途 |
|------|------|------|----------|
| FP32 | 1x | 高 | 科学计算 |
| TF32 | 8x | 中 | AI 训练 |
| FP16 | 16x | 低 | AI 推理 |

### 4. 使用 Stream

```cuda
cublasSetStream(handle, stream);
```

## 实战：完整的 GEMM 示例

```cuda
#include <cublas_v2.h>
#include <stdio.h>

#define M 1024
#define K 1024
#define N 1024

void verify_gemm() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // 分配内存
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_a);
    float *h_B = (float*)malloc(size_b);
    float *h_C = (float*)malloc(size_c);

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = i * 0.01f;
    for (int i = 0; i < K * N; i++) h_B[i] = i * 0.01f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_a);
    cudaMalloc(&d_B, size_b);
    cudaMalloc(&d_C, size_c);

    cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 执行 GEMM
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha, d_A, M,
                d_B, K,
                &beta, d_C, M);

    // 拷贝结果
    cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost);

    // 验证（打印部分结果）
    printf("C[0:4, 0:4]:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%10.4f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cublasDestroy(handle);
}
```

## 练习

### TODO: 完成 cublas_example.cu

1. 初始化 cuBLAS handle
2. 实现基本的 GEMM
3. 验证结果正确性

### 进阶练习

1. 实现批量 GEMM
2. 比较 cuBLAS 和手写 kernel 性能
3. 使用 stridedBatched GEMM

## 调试

### 错误处理

```cuda
cublasStatus_t status = cublasSgemm(...);
if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS error: %d\n", status);
}
```

### 常见错误码

| 错误码 | 值 | 含义 |
|--------|-----|------|
| CUBLAS_STATUS_SUCCESS | 0 | 成功 |
| CUBLAS_STATUS_NOT_INITIALIZED | 1 | handle 未初始化 |
| CUBLAS_STATUS_ALLOC_FAILED | 3 | 内存分配失败 |
| CUBLAS_STATUS_INVALID_VALUE | 7 | 参数无效 |
| CUBLAS_STATUS_ARCH_MISMATCH | 11 | 架构不支持 |

## 常见问题

### Q1: cuBLAS 和 cuDNN 有什么区别？

- cuBLAS: 基础线性代数
- cuDNN: 深度学习专用（Conv, Pooling, RNN 等）

### Q2: 如何选择 leading dimension？

```
lda >= rows_of_A (列优先)
```

通常 `lda = rows`，但有时为了内存对齐会设置更大。

### Q3: 性能不如预期？

检查：
1. 是否启用了 Tensor Core
2. 矩阵维度是否对齐
3. 内存带宽是否瓶颈

## 下一步

完成 [cublas_example.cu](../tutorials/08_cublas/cublas_example.cu) 的 TODO 部分，然后学习 [Lesson 09: 性能优化](09_performance_optimization.md)。
