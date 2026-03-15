# Lesson 05: CUDA Streams 和 Events

本教程介绍 CUDA 流和事件，实现并发执行和精确计时。

## 学习目标

- 理解 CUDA Stream 的概念
- 掌握多流并发执行技术
- 学习使用 Events 进行精确计时
- 了解异步内存传输

## 文件结构

```
05_streams/
├── streams.cu              # TODO 练习代码
├── solution/
│   └── streams.cu          # 参考答案
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
- `tutorials/05_streams/streams` - TODO 版本
- `tutorials/05_streams/streams_solution` - 参考答案

### 方法二：使用 nvcc 单独编译

```bash
cd tutorials/05_streams

# 编译 TODO 版本
nvcc -o streams streams.cu -I../../include

# 编译参考答案
nvcc -o streams_solution solution/streams.cu -I../../include
```

## 运行方法

```bash
# 运行 TODO 版本
./streams

# 运行参考答案
./streams_solution
```

### 预期输出

```
=== Lesson 05: Streams 和 Events ===

GPU: NVIDIA GeForce RTX 3060 Laptop GPU
计算能力：8.6

测试 1: 默认 stream (顺序执行)
--------------------------------
时间：XX.XXX ms

测试 2: 多个 streams (并发执行)
--------------------------------
时间：XX.XXX ms

测试 3: Events 精确计时
----------------------
Kernel 执行时间 (Event): X.XXX ms
```

## 练习任务

1. **创建多个 Streams**：使用 `cudaStreamCreate`
2. **并发执行**：在不同 stream 中执行 kernel 和内存拷贝
3. **使用 Events 计时**：记录 kernel 执行时间
4. **对比性能**：比较顺序执行和并发执行的时间

## 关键概念

### 什么是 CUDA Stream？

Stream 是一系列按顺序执行的 CUDA 操作（kernel、内存拷贝等）。不同 stream 中的操作可以并发执行。

```
Stream 0: [H2D] -> [Kernel] -> [D2H]
Stream 1: [H2D] -> [Kernel] -> [D2H]
Stream 2: [H2D] -> [Kernel] -> [D2H]
Stream 3: [H2D] -> [Kernel] -> [D2H]

时间轴：
Stream 0: ====####++++
Stream 1:   ====####++++
Stream 2:     ====####++++
Stream 3:       ====####++++
```

### Stream API

```cpp
// 创建 stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 在 stream 中执行 kernel
kernel_name<<<grid, block, shared_mem, stream>>>(args);

// 在 stream 中进行内存拷贝
cudaMemcpyAsync(dst, src, size, kind, stream);

// 等待 stream 完成
cudaStreamSynchronize(stream);

// 销毁 stream
cudaStreamDestroy(stream);
```

### Events API

```cpp
// 创建 events
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 记录开始
cudaEventRecord(start, stream);

// 执行操作...

// 记录结束
cudaEventRecord(stop, stream);

// 等待 event 完成
cudaEventSynchronize(stop);

// 获取经过的时间（毫秒）
float elapsed;
cudaEventElapsedTime(&elapsed, start, stop);

// 销毁 events
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 异步内存拷贝

```cpp
// 使用 stream 进行异步拷贝
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);

// 注意：必须配合 cudaStreamSynchronize 或 cudaDeviceSynchronize 使用
```

## 练习答案提示

<details>
<summary>多流并发执行示例</summary>

```cpp
void multi_stream_example(float* h_a, float* h_b, int n) {
    const int NUM_STREAMS = 4;
    int elements_per_stream = n / NUM_STREAMS;
    size_t size_per_stream = elements_per_stream * sizeof(float);

    cudaStream_t streams[NUM_STREAMS];
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];

    // 1. 创建多个 streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 2. 为每个 stream 分配 device 内存
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_a[i], size_per_stream);
        cudaMalloc(&d_b[i], size_per_stream);
        cudaMalloc(&d_c[i], size_per_stream);
    }

    // 3. 在每个 stream 中执行操作
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * elements_per_stream;

        // 异步 H2D 拷贝
        cudaMemcpyAsync(d_a[i], h_a + offset, size_per_stream,
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, size_per_stream,
                       cudaMemcpyHostToDevice, streams[i]);

        // 启动 kernel（指定 stream）
        int blocks = (elements_per_stream + 255) / 256;
        vector_add<<<blocks, 256, 0, streams[i]>>>(
            d_a[i], d_b[i], d_c[i], elements_per_stream);

        // 异步 D2H 拷贝
        cudaMemcpyAsync(h_c + offset, d_c[i], size_per_stream,
                       cudaMemcpyDeviceToHost, streams[i]);
    }

    // 4. 等待所有 streams 完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
    }
}
```

</details>

## 常见问题

**Q: Stream 0 有什么特殊之处？**
A: Stream 0 是默认 stream，其他 stream 中的操作会与 stream 0 同步。

**Q: 如何判断 GPU 是否支持并发执行？**
A: 使用 `cudaDeviceGetStreamPriorityRange` 查询流优先级范围，或查看 GPU 是否支持 Concurrent Kernel Execution。

**Q: Events 和 Streams 有什么区别？**
A: Stream 是操作的队列，Event 是时间点的标记，用于同步和计时。

**Q: 异步拷贝和同步拷贝有什么区别？**
A: 异步拷贝立即返回，不等待拷贝完成；同步拷贝会阻塞直到完成。

## 性能提示

1. **使用非阻塞流**：避免在流之间不必要的同步
2. **重叠传输和计算**：在一个流传输数据时，另一个流执行计算
3. **合理分配工作**：确保每个流有足够的工作量来利用并发

## 下一步

完成本教程后，继续学习：
- [Lesson 06: CUDA Graph](../06_cuda_graph/README.md)
