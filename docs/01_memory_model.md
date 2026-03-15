# Lesson 01: 内存模型

## 学习目标

- 理解 Host/Device 内存架构
- 掌握 CUDA 内存管理 API
- 学习数据在 CPU 和 GPU 之间传输
- 了解统一内存（Unified Memory）

## GPU 内存架构

### 分离式内存

```
┌─────────────┐         ┌─────────────┐
│    CPU      │         │     GPU     │
│   Host      │   PCI   │   Device    │
│   Memory    │←───────→│   Memory    │
│  (DDR4/5)   │  Express│  (GDDR6X)   │
└─────────────┘         └─────────────┘
```

### 内存带宽对比

| 类型 | 带宽 | 延迟 |
|------|------|------|
| DDR5 | ~50 GB/s | ~50ns |
| GDDR6X | ~1000 GB/s | ~200ns |
| HBM2e | ~3000 GB/s | ~100ns |

## 内存管理 API

### 1. cudaMalloc

```cuda
float* d_data;
size_t size = N * sizeof(float);

// 分配 device 内存
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    // 处理错误
}
```

### 2. cudaMemcpy

```cuda
// Host → Device
cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);

// Device → Device
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
```

### 3. cudaMemset

```cuda
// 初始化 device 内存（按字节）
cudaMemset(d_data, 0, size);
```

### 4. cudaFree

```cuda
cudaFree(d_data);
```

## 向量加法示例

### 完整代码

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    // 分配 host 内存
    float *h_a = malloc(size);
    float *h_b = malloc(size);
    float *h_c = malloc(size);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // 分配 device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 拷贝到 device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 启动 kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // 拷贝回 host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
```

### 执行流程

```
1. malloc 分配 host 内存
2. 初始化数据
3. cudaMalloc 分配 device 内存
4. cudaMemcpy H2D
5. kernel 执行
6. cudaMemcpy D2H
7. cudaFree 释放
```

## 统一内存（Unified Memory）

### 什么是统一内存？

统一内存在 CUDA 6.0 引入，简化了内存管理。

```cuda
float* data;

// 统一内存分配
cudaMallocManaged(&data, size);

// 直接在 kernel 中使用
kernel<<<blocks, threads>>>(data, n);

// 访问数据（自动同步）
cudaDeviceSynchronize();
for (int i = 0; i < n; i++) {
    printf("%f\n", data[i]);  // 直接访问！
}

cudaFree(data);
```

### 优缺点对比

| 特性 | 传统内存 | 统一内存 |
|------|----------|----------|
| 编程复杂度 | 高 | 低 |
| 性能 | 最优 | 略低 |
| 内存超订阅 | 不支持 | 支持 |
| 控制粒度 | 细 | 粗 |

### 内存迁移

```
CPU 访问 → 页面迁移到 CPU
GPU 访问 → 页面迁移到 GPU
```

## 内存传输优化

### 1. 使用 pinned memory

```cuda
float* h_data;
// 分配 pinned memory（page-locked）
cudaMallocHost(&h_data, size);

// 传输更快，可异步
cudaMemcpyAsync(d_data, h_data, size, ...);

cudaFreeHost(h_data);
```

### 2. 异步传输

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_data, h_data, size,
                cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(d_data);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

### 3. 零拷贝内存

```cuda
float* zc_data;
cudaHostAlloc(&zc_data, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_zc_data, zc_data, 0);

// 直接访问，但较慢
kernel<<<blocks, threads>>>(d_zc_data);
```

## 带宽测量

```cuda
SimpleTimer timer;
timer.startRecord();
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
timer.stopRecord();

float bandwidth = size / (timer.elapsedMs() * 1e-6) / 1e9;
printf("带宽：%.2f GB/s\n", bandwidth);
```

典型值：
- PCIe 3.0 x16: ~12 GB/s
- PCIe 4.0 x16: ~24 GB/s

## 练习

### TODO: 完成 vector_add.cu

1. 补全内存分配代码
2. 实现 kernel 函数
3. 验证结果正确性

### 进阶练习

1. 使用统一内存重写程序
2. 比较两种方式的性能差异
3. 测量内存传输带宽

## 常见问题

### Q1: cudaMalloc 失败怎么办？

```cuda
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("可用内存：%zu MB\n", free_mem / 1024 / 1024);
```

### Q2: 内存泄漏检测

```bash
cuda-memcheck ./your_program
```

### Q3: 传输太慢？

- 检查是否使用 pinned memory
- 确认 PCIe 链路状态
- 考虑使用统一内存

## 下一步

完成 [vector_add.cu](../tutorials/01_vector_add/vector_add.cu) 的 TODO 部分，然后学习 [Lesson 02: 线程层次结构](02_thread_hierarchy.md)。
