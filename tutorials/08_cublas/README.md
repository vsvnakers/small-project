# Lesson 08: cuBLAS 库

本教程介绍 NVIDIA cuBLAS 库的使用，进行高性能线性代数运算。

## 学习目标

- 了解 cuBLAS 库的基本概念
- 掌握 GEMM（通用矩阵乘法）API
- 理解列优先存储和行优先存储的转换
- 学习批量矩阵运算

## 文件结构

```
08_cublas/
├── cublas_example.cu       # TODO 练习代码
├── solution/
│   └── cublas_example.cu   # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

```bash
cd /mnt/e/small-project
mkdir -p build && cd build
cmake ..
make -j
```

可执行文件位于：
- `tutorials/08_cublas/cublas_example` - TODO 版本
- `tutorials/08_cublas/cublas_example_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/08_cublas

# 编译 TODO 版本（需要链接 cuBLAS）
nvcc -o cublas_example cublas_example.cu -I../../include -lcublas

# 编译参考答案
nvcc -o cublas_example_solution solution/cublas_example.cu -I../../include -lcublas
```

## 运行方法

```bash
# 运行 TODO 版本
./cublas_example

# 运行参考答案
./cublas_example_solution
```

### 预期输出

```
=== Lesson 08: cuBLAS 库 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

矩阵 A: 1024 x 1024
矩阵 B: 1024 x 1024
矩阵 C: 1024 x 1024

测试：cuBLAS GEMM
---------------
cuBLAS 时间：X.XXX ms

结果示例 (C[0:4, 0:4]):
   0.000    0.000    0.000    0.000
   0.000    0.000    0.000    0.000
   0.000    0.000    0.000    0.000
   0.000    0.000    0.000    0.000
```

## 练习任务

1. **创建 cuBLAS Handle**：初始化 cuBLAS 上下文
2. **实现 GEMM**：调用 `cublasSgemm` 进行矩阵乘法
3. **理解存储顺序**：处理列优先和行优先的差异
4. **清理资源**：销毁 cuBLAS handle

## 关键概念

### 什么是 cuBLAS？

cuBLAS 是 NVIDIA 提供的 GPU 加速基本线性代数子程序库，提供高度优化的矩阵和向量运算。

### cuBLAS API 层次

| 层级 | 函数 | 描述 |
|------|------|------|
| Level 1 | `cublasSaxpy`, `cublasSdot` | 向量 - 向量运算 |
| Level 2 | `cublasSgemv` | 矩阵 - 向量运算 |
| Level 3 | `cublasSgemm` | 矩阵 - 矩阵运算 |

### GEMM 公式

```
C = alpha * op(A) * B + beta * C

其中：
- alpha, beta: 缩放因子
- op(A): A 或 A 的转置
- A: m × k 矩阵
- B: k × n 矩阵
- C: m × n 矩阵
```

### 列优先 vs 行优先

**重要**：cuBLAS 使用列优先存储（column-major），而 C/C++ 使用行优先存储（row-major）。

```
行优先 (C/C++):          列优先 (cuBLAS/Fortran):
[ 0  1  2 ]              [ 0  3  6 ]
[ 3  4  5 ]    内存：     [ 1  4  7 ]
[ 6  7  8 ]    → 0,1,2,3,4,5,6,7,8
```

### 核心 API

```cpp
// 1. 创建 handle
cublasHandle_t handle;
cublasCreate(&handle);

// 2. 执行 GEMM（单精度）
// C = A * B (行优先) 等价于 C^T = B^T * A^T (列优先)
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
            n, m, k,                   // 注意：列优先下 n 和 m 交换
            &alpha,
            d_B, n,                    // B 是 k x n (列优先)
            d_A, m,                    // A 是 m x k (列优先)
            &beta,
            d_C, n);                   // C 是 n x m (列优先)

// 3. 设置数学模式（启用 Tensor Core）
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// 4. 销毁 handle
cublasDestroy(handle);
```

### 常用 GEMM 函数

```cpp
// 单精度 GEMM
cublasSgemm(handle, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

// 双精度 GEMM
cublasDgemm(handle, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

// 半精度 GEMM（支持 Tensor Core）
cublasGemmEx(handle, ...);

// 批量 GEMM
cublasSgemmBatched(handle, ...);
```

## 练习答案提示

<details>
<summary>cuBLAS GEMM 完整实现</summary>

```cpp
void cublas_gemm(cublasHandle_t handle,
                 float* d_A, float* d_B, float* d_C,
                 int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 使用列优先存储
    // C = A * B (行优先) 等价于 C^T = B^T * A^T (列优先)
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // 转置输入
        n, m, k,
        &alpha,
        d_B, k,      // B 是 k x n, 转置后为 n x k
        d_A, m,      // A 是 m x k, 转置后为 k x m
        &beta,
        d_C, n       // C 是 m x n, 结果为 n x m
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM 失败：%d\n", status);
        exit(EXIT_FAILURE);
    }
}
```

```cpp
// 主函数中的使用
int main() {
    // 创建 cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // ... 分配内存 ...

    // 执行 GEMM
    cublas_gemm(cublas_handle, d_A, d_B, d_C, M, N, K);

    // 清理
    cublasDestroy(cublas_handle);
}
```

</details>

## 常见问题

**Q: 为什么 cuBLAS 结果和 CPU 结果不一致？**
A: 可能是存储顺序问题。cuBLAS 使用列优先，需要调整参数或转置矩阵。

**Q: cublasSgemm 和 cublasDgemm 有什么区别？**
A: `Sgemm` 是单精度（float），`Dgemm` 是双精度（double）。

**Q: 如何启用 Tensor Core 加速？**
A: 使用 `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)` 和 `cublasGemmEx` API。

**Q: 什么是 LDA（Leading Dimension）？**
A: LDA 是矩阵在内存中的实际行数，通常等于矩阵的行数，但在使用子矩阵时可能不同。

## 性能提示

1. **使用合适的精度**：FP16 比 FP32 快 8-16 倍（有 Tensor Core 时）
2. **批量运算**：使用 `cublasGemmBatched` 处理多个小矩阵
3. **避免不必要的拷贝**：直接在 device 内存上操作
4. **复用 handle**：创建一次 handle，重复使用

## 下一步

完成本教程后，继续学习：
- [Lesson 09: 性能优化](../09_optimization/README.md)
