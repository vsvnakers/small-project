# Lesson 09: 性能优化

## 学习目标

- 理解 CUDA 性能瓶颈
- 学习内存合并访问
- 掌握 Occupancy 优化
- 使用 Profiling 工具分析

## 性能瓶颈分析

### Roofline 模型

```
性能 (FLOPS)
    ↑
    │     ┌──────────── 峰值计算能力
    │    /
    │   /
    │  /
    │ /
    │/ ←── 内存带宽限制
    └──────────────────→ 算术强度 (FLOPS/Byte)
```

### 瓶颈类型

| 类型 | 特征 | 优化方向 |
|------|------|----------|
| 内存受限 | 低算术强度 | 合并访问、共享内存 |
| 计算受限 | 高算术强度 | 指令级优化、Tensor Core |
| 延迟受限 | 小数据量 | Occupancy、并行度 |

## 内存合并访问

### 什么是合并访问？

当相邻线程访问相邻内存地址时，多个访问可以合并为一次事务。

```
合并访问 (✓):          非合并访问 (✗):
Thread 0 → Address 0    Thread 0 → Address 0
Thread 1 → Address 1    Thread 1 → Address 8
Thread 2 → Address 2    Thread 2 → Address 16
Thread 3 → Address 3    Thread 3 → Address 100
     ↓                       ↓
一次 128-byte 事务        四次独立事务
```

### 优化示例

```cuda
// 错误：随机访问
__global__ void bad_access(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int random_idx = (idx * 7919) % n;  // 质数乘法
    output[idx] = input[random_idx];    // 非合并
}

// 正确：顺序访问
__global__ void good_access(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx];  // 合并访问
}
```

### 性能对比

| 访问模式 | 带宽 | 相对性能 |
|----------|------|----------|
| 合并访问 | ~900 GB/s | 100% |
| 随机访问 | ~100 GB/s | 11% |

## 向量化内存访问

### 使用 float4

```cuda
__global__ void vectorized_copy(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // float4 向量化
    int float4_idx = idx * 4;
    while (float4_idx + 3 < n) {
        float4 in = reinterpret_cast<const float4*>(&input[float4_idx])[0];
        reinterpret_cast<float4*>(&output[float4_idx])[0] = in;
        float4_idx += stride * 4;
    }

    // 处理剩余
    for (int i = float4_idx; i < n; i++) {
        output[i] = input[i];
    }
}
```

### 性能提升

| 类型 | 指令数 | 带宽 |
|------|--------|------|
| float | 4×load + 4×store | 基准 |
| float4 | 1×load + 1×store | 3-4x |

## Occupancy 优化

### 什么是 Occupancy？

Occupancy 是 SM 上活跃的 warp 数量与最大可能数量的比值。

```
Occupancy = 活跃 warp 数 / 最大 warp 数
```

### 限制因素

| 资源 | Volta | Ampere |
|------|-------|--------|
| 每 SM 最大线程数 | 2048 | 2048 |
| 每 SM 最大 block 数 | 32 | 32 |
| 每 SM 共享内存 | 64 KB | 64 KB |
| 每线程寄存器 | 255 | 255 |

### 查询最优配置

```cuda
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
                                   kernel, 0, 0);
printf("最优 block size: %d\n", block_size);
```

### 优化策略

```cuda
// 1. 减少寄存器使用
__launch_bounds__(256, 2)  // 限制寄存器，增加 occupancy
__global__ void kernel() {
    ...
}

// 2. 调整 block size
// 尝试 128, 256, 512

// 3. 使用 grid-stride loop
__global__ void grid_stride(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        data[i] = ...;
    }
}
```

## 共享内存优化

### Bank Conflict 避免

```cuda
// 错误：32-way bank conflict
__shared__ float sdata[256];
for (int i = 0; i < 32; i++) {
    sum += sdata[i * 32];  // 都访问 bank 0
}

// 正确：添加 padding
__shared__ float sdata[256 + 8];
for (int i = 0; i < 32; i++) {
    sum += sdata[i * 33];  // 分散到不同 bank
}
```

### 优化归约

```cuda
// 优化 1：无分支
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

// 优化 2：warp 级优化
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
// warp 内隐式同步
if (tid < 32) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

## Profiling 工具

### Nsight Systems

```bash
# 生成报告
nsys profile --stats=true ./your_program

# 生成 trace 文件
nsys profile -o trace ./your_program
```

### Nsight Compute

```bash
# kernel 详细分析
ncu ./your_program

# 特定指标
ncu --metrics sm__throughput ./your_program
```

### 关键指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Occupancy | 活跃 warp 比例 | >50% |
| Memory Throughput | 内存吞吐 | 接近峰值 |
| Compute Throughput | 计算吞吐 | 接近峰值 |
| Branch Efficiency | 分支效率 | >90% |

## 实战：优化矩阵乘法

### 版本对比

```cuda
// V1: 基础版本
__global__ void matmul_v1(float* A, float* B, float* C, int n) {
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

// V2: 共享内存
__global__ void matmul_v2(float* A, float* B, float* C, int n) {
    __shared__ float As[16][16], Bs[16][16];
    // ... 加载到共享内存，计算
}

// V3: 向量化加载
__global__ void matmul_v3(float* A, float* B, float* C, int n) {
    // 使用 float4 加载
}

// V4: Tensor Core
__global__ void matmul_v4(float* A, float* B, float* C, int n) {
    // 使用 WMMA API
}
```

### 性能对比 (RTX 4090, 1024×1024)

| 版本 | 时间 | 加速 |
|------|------|------|
| V1 基础 | 8.5 ms | 1x |
| V2 共享内存 | 2.1 ms | 4x |
| V3 向量化 | 1.5 ms | 5.7x |
| V4 Tensor Core | 0.3 ms | 28x |

## 练习

### TODO: 完成 optimization.cu

1. 实现合并访问版本
2. 使用 profiler 分析性能
3. 尝试不同优化策略

### 进阶练习

1. 实现完整的矩阵乘法优化
2. 比较不同 block size 的影响
3. 使用 Tensor Core

## 调试技巧

### 内存带宽测量

```cuda
float bandwidth = (bytes_read + bytes_written) / (time_ms * 1e-6) / 1e9;
printf("带宽：%.2f GB/s\n", bandwidth);
```

### Occupancy 检查

```cuda
int active_blocks, active_warps;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                               kernel, block_size, 0);
printf("每 SM 活跃 block: %d\n", active_blocks);
```

## 常见问题

### Q1: Occupancy 越高越好吗？

不一定。有时降低 occupancy 可以减少寄存器压力，提高单线程性能。

### Q2: 如何确定瓶颈？

使用 Nsight Compute:
```bash
ncu --metrics pipelining__l1tex__data_pipe_averaged_latency_pct ./your_program
```

### Q3: 共享内存越大越好吗？

太大可能减少 occupancy。平衡使用。

## 总结

优化检查清单：

- [ ] 内存访问合并
- [ ] 使用向量化加载
- [ ] 共享内存优化
- [ ] 避免 bank conflict
- [ ] Occupancy 合理
- [ ] 使用 Tensor Core（如适用）
- [ ] 使用 Stream 并发

恭喜完成整个 CUDA 教程！
