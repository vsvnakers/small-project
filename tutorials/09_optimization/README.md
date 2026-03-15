# Lesson 09: 性能优化

本教程介绍 CUDA 性能优化技巧和 profiling 工具的使用。

## 学习目标

- 理解内存合并访问（Coalesced Access）
- 掌握 Occupancy 优化技术
- 学习向量化内存访问
- 使用 profiling 工具分析性能瓶颈

## 文件结构

```
09_optimization/
├── optimization.cu         # TODO 练习代码
├── solution/
│   └── optimization.cu     # 参考答案
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
- `tutorials/09_optimization/optimization` - TODO 版本
- `tutorials/09_optimization/optimization_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd /mnt/e/small-project/tutorials/09_optimization

# 编译 TODO 版本
nvcc -o optimization optimization.cu -I../../include

# 编译参考答案
nvcc -o optimization_solution solution/optimization.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./optimization

# 运行参考答案
./optimization_solution
```

### 预期输出

```
=== Lesson 09: 性能优化 ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

数据大小：64.00 MB
线程配置：65536 blocks x 256 threads

性能测试结果:
=============================
未优化 (随机)       :  XX.XXX ms,  XXX.XX GB/s
优化 (顺序)         :  XX.XXX ms,  XXX.XX GB/s
共享内存            :  XX.XXX ms,  XXX.XX GB/s
向量化 (float4)     :  XX.XXX ms,  XXX.XX GB/s
Occupancy 优化      :  XX.XXX ms,  XXX.XX GB/s
=============================
```

## 练习任务

1. **实现顺序访问**：将随机访问改为合并访问
2. **优化 Occupancy**：使用 occupancy API 查询最优配置
3. **实现向量化访问**：使用 float4 类型
4. **性能对比**：比较不同优化技术的效果

## 关键概念

### 内存合并访问（Coalesced Access）

**合并访问**：相邻线程访问相邻的内存地址，可以合并为少数几次内存事务。

```
合并访问（高效）：
Thread 0: addr[0]
Thread 1: addr[1]
Thread 2: addr[2]
Thread 3: addr[3]
→ 1 次内存事务

随机访问（低效）：
Thread 0: addr[100]
Thread 1: addr[5000]
Thread 2: addr[23]
Thread 3: addr[9999]
→ 4 次内存事务
```

### Occupancy

**Occupancy**：SM 上活跃的 warp 数量与最大可能 warp 数量的比值。

高 occupancy 可以隐藏内存延迟，但不是越高越好。

```cpp
int min_grid_size, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                   kernel, 0, 0);

int active_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                               kernel, block_size, 0);
```

### 向量化内存访问

使用向量类型（如 `float4`）减少内存指令数量：

```cpp
// 普通访问：4 条 load 指令
float a = input[i];
float b = input[i+1];
float c = input[i+2];
float d = input[i+3];

// 向量化：1 条 load 指令
float4 v = reinterpret_cast<const float4*>(&input[i])[0];
```

### 优化技巧对比

| 技术 | 适用场景 | 预期提升 |
|------|----------|----------|
| 合并访问 | 所有 kernel | 2-10x |
| Occupancy 优化 | 内存密集型 | 10-30% |
| 向量化访问 | 简单数据处理 | 2-4x |
| 共享内存 | 数据复用场景 | 2-8x |
| 减少分支 | 条件执行多的 kernel | 10-50% |

## 练习答案提示

<details>
<summary>优化的 kernel 实现</summary>

```cpp
// 顺序访问（合并访问）
__global__ void optimized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 顺序访问：相邻线程访问相邻地址
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// 向量化访问
__global__ void vectorized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int float4_idx = idx * 4;
    while (float4_idx + 3 < n) {
        // 向量化加载
        float4 in = reinterpret_cast<const float4*>(&input[float4_idx])[0];
        float4 out;
        out.x = in.x * 2.0f + 1.0f;
        out.y = in.y * 2.0f + 1.0f;
        out.z = in.z * 2.0f + 1.0f;
        out.w = in.w * 2.0f + 1.0f;
        // 向量化存储
        reinterpret_cast<float4*>(&output[float4_idx])[0] = out;
        float4_idx += stride * 4;
    }

    // 处理剩余元素
    for (int i = float4_idx; i < n; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}
```

</details>

## Profiling 工具

### Nsight Systems

时间线分析工具，查看 kernel 执行顺序和并发情况：

```bash
nsys profile --stats=true ./optimization_solution
```

### Nsight Compute

详细的 kernel 性能分析：

```bash
ncu --metrics achieved__occupancy,speed_of_light_throughput \
    ./optimization_solution
```

### nvprof（旧版）

```bash
nvprof ./optimization_solution
```

## 常见问题

**Q: 为什么随机访问性能差？**
A: 随机访问导致内存事务无法合并，每次访问都可能需要单独的内存请求。

**Q: Occupancy 越高越好吗？**
A: 不一定。有时减少寄存器使用来提高 occupancy，反而可能降低性能。

**Q: 如何选择合适的 Block Size？**
A: 通常是 32 的倍数（128, 256, 512）。使用 `cudaOccupancyMaxPotentialBlockSize` 查询。

**Q: 共享内存总是更好吗？**
A: 不是。共享内存有容量限制，且不当使用会导致 Bank Conflict。

## 性能优化检查清单

- [ ] 内存访问是否合并？
- [ ] 是否有不必要的分支？
- [ ] 共享内存是否有 Bank Conflict？
- [ ] Occupancy 是否足够？
- [ ] 是否使用了向量化访问？
- [ ] 是否减少了 Host-Device 传输？
- [ ] 是否使用了异步拷贝？
- [ ] 是否分析了性能瓶颈？

## 下一步

完成本教程后，你已经掌握了 CUDA 编程的核心技能！

建议继续学习：
- CUDA C Programming Guide
- NVIDIA Developer Blog
- cuBLAS、cuDNN 等高性能库
- 实际项目实践

恭喜完成整个 CUDA 教程系列！
