# Lesson 02: 线程层次结构

## 学习目标

- 理解 Thread、Block、Grid 的概念
- 掌握多维线程索引计算
- 学习二维和三维线程块配置
- 实现矩阵乘法

## 线程层次结构

### 三级结构

```
Grid (kernel 启动)
├── Block (0, 0)
│   ├── Thread (0, 0)
│   ├── Thread (0, 1)
│   ├── Thread (1, 0)
│   └── Thread (1, 1)
├── Block (1, 0)
│   └── ...
└── ...
```

### 内置变量

| 变量 | 含义 | 范围 |
|------|------|------|
| `threadIdx` | 线程在 block 内的索引 | 0 ~ 1023 |
| `blockIdx` | block 在 grid 内的索引 | 0 ~ 65535 |
| `blockDim` | block 的维度大小 | - |
| `gridDim` | grid 的维度大小 | - |

## 一维线程配置

### 简单索引计算

```cuda
// 配置：256 个线程/block
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### Grid-stride Loop

```cuda
__global__ void grid_stride(float* data, int n) {
    // 计算全局线程 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算 grid 总大小
    int stride = blockDim.x * gridDim.x;

    // 每个线程处理多个元素
    for (int i = idx; i < n; i += stride) {
        data[i] = ...;
    }
}
```

## 二维线程配置

### 适用场景

- 图像处理
- 矩阵运算
- 2D 网格模拟

### 索引计算

```cuda
__global__ void kernel_2d(float* matrix, int width, int height) {
    // 计算二维坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (x < width && y < height) {
        // 转换为线性索引（行优先）
        int idx = y * width + x;
        matrix[idx] = ...;
    }
}
```

### 启动配置

```cuda
dim3 threads(16, 16);  // 16x16 = 256 线程/block
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel_2d<<<grid, threads>>>(...);
```

## 三维线程配置

### 适用场景

- 3D 图像处理
- 体积数据
- 张量运算

### 索引计算

```cuda
__global__ void kernel_3d(float* volume, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int idx = (z * H + y) * W + x;
        volume[idx] = ...;
    }
}
```

## 矩阵乘法实现

### 基础版本

```cuda
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

### 计算复杂度

- 每个元素：O(n) 次乘加
- 总计：O(n³) 次操作
- 并行度：O(n²) 个线程

## Block 配置优化

### 选择 Block Size

考虑因素：

1. **Occupancy**: 256 或 512 通常是好选择
2. **共享内存**: 每个 block 的共享内存有限
3. **寄存器**: 每个 thread 的寄存器使用

### 推荐配置

| GPU 架构 | 推荐 Block Size |
|----------|-----------------|
| Pascal | 128, 256 |
| Volta+ | 256, 512 |
| Ampere | 256, 512 |

### 查询最优配置

```cuda
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
                                   kernel, 0, 0);
```

## 实践：矩阵乘法

### 完整代码

```cuda
#define BLOCK_SIZE 16

__global__ void matrix_mul(float* A, float* B, float* C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 分块计算
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 加载数据到共享内存
        if (row < n && tile * BLOCK_SIZE + tx < n)
            As[ty][tx] = A[row * n + tile * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (tile * BLOCK_SIZE + ty < n && col < n)
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}
```

## 练习

### TODO: 完成 matrix_mul.cu

1. 实现基础版本（不使用共享内存）
2. 计算正确的二维索引
3. 验证结果

### 进阶练习

1. 实现共享内存优化版本
2. 比较两种实现的性能
3. 尝试不同的 block size

## 调试技巧

### 打印线程信息

```cuda
__global__ void debug_kernel() {
    printf("Block: (%d, %d), Thread: (%d, %d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y);
}
```

### 检查边界

```cuda
if (idx >= n) {
    printf("Error: idx=%d, n=%d\n", idx, n);
    return;
}
```

## 下一步

完成 [matrix_mul.cu](../tutorials/02_matrix_mul/matrix_mul.cu) 的 TODO 部分，然后学习 [Lesson 03: 共享内存](03_shared_memory.md)。
