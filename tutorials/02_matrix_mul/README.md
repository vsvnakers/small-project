# Lesson 02: 矩阵乘法

本教程介绍二维线程块和共享内存优化技术。

## 学习目标

- 掌握二维线程索引计算
- 理解矩阵乘法的并行化方法
- 学习共享内存的使用
- 了解内存访问模式对性能的影响

## 文件结构

```
02_matrix_mul/
├── matrix_mul.cu           # TODO 练习代码
├── solution/
│   └── matrix_mul.cu       # 参考答案
├── CMakeLists.txt          # 构建配置
└── README.md               # 本文件
```

## 编译方法

### 方法一：使用 CMake（推荐）

```bash
cd <project-root>
mkdir -p build && cd build
cmake ..
make -j
```

其中 `<project-root>` 是你克隆项目后的根目录。

可执行文件位于：
- `tutorials/02_matrix_mul/matrix_mul` - TODO 版本
- `tutorials/02_matrix_mul/matrix_mul_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd tutorials/02_matrix_mul

# 编译 TODO 版本
nvcc -o matrix_mul matrix_mul.cu -I../../include

# 编译参考答案
nvcc -o matrix_mul_solution solution/matrix_mul.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./matrix_mul

# 运行参考答案
./matrix_mul_solution
```

### 预期输出

```
=== Lesson 02: 矩阵乘法 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

矩阵大小：512 x 512
线程块：16x16
Grid: 32x32

基础版本 GPU 时间：XX.XXX ms
CPU 参考时间：XXX.XXX ms

验证结果：GPU 和 CPU 结果匹配 ✓
```

## 练习任务

1. **实现基础版本**：完成 `matrix_mul_basic` kernel
2. **理解二维索引**：计算正确的行列索引
3. **实现共享内存版本**：完成 `matrix_mul_shared` kernel
4. **性能对比**：比较两个版本的执行时间

## 关键概念

### 二维线程索引

```cpp
// 计算二维线程 ID
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
```

### 二维 Grid 和 Block 配置

```cpp
dim3 threads(BLOCK_SIZE, BLOCK_SIZE);  // 16x16 线程块
dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
          (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

### 矩阵乘法公式

```
C[i][j] = Σ A[i][k] * B[k][j]  (k 从 0 到 N-1)
```

### 共享内存声明和使用

```cpp
// 声明共享内存（block 级别共享）
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

// 加载数据到共享内存
As[threadIdx.y][threadIdx.x] = A[row * N + col];

// 同步 block 内所有线程
__syncthreads();
```

## 练习答案提示

<details>
<summary>基础版本答案</summary>

```cpp
__global__ void matrix_mul_basic(float* A, float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
```

</details>

<details>
<summary>共享内存版本思路</summary>

1. 将矩阵分成小块（tile）加载到共享内存
2. 每个线程块协作加载一个 tile
3. 同步后从共享内存计算
4. 重复直到处理完所有 k

</details>

## 常见问题

**Q: 为什么要用二维线程块？**
A: 矩阵是二维结构，二维线程块更自然地映射到问题域，每个线程负责一个输出元素。

**Q: blockDim.x 和 blockDim.y 有什么区别？**
A: `blockDim.x` 是 block 在 x 方向的线程数，`blockDim.y` 是 y 方向的线程数。

**Q: 共享内存有什么好处？**
A: 共享内存比全局内存快得多，可以复用加载的数据，减少全局内存访问。

**Q: __syncthreads() 的作用是什么？**
A: 同步 block 内所有线程，确保所有线程都完成数据加载后再进行计算。

## 性能优化技巧

1. **共享内存分块**：将数据分块加载到共享内存
2. **减少 Bank Conflict**：合理设计共享内存访问模式
3. **寄存器使用**：使用局部变量存储中间结果
4. **合并内存访问**：确保相邻线程访问相邻内存地址

## 下一步

完成本教程后，继续学习：
- [Lesson 03: 共享内存归约](../03_shared_memory/README.md)
