# Lesson 05: Streams & Events

## 学习目标

- 理解 CUDA Stream 的概念
- 学习使用多个 Stream 实现并发
- 掌握 CUDA Events 进行计时和同步
- 实现异步内存传输

## CUDA Stream 基础

### 什么是 Stream？

Stream 是 GPU 上的一个**操作队列**，操作按顺序执行，不同 stream 可以并发。

```
Stream 0 (default): [H2D] → [Kernel] → [D2H]
Stream 1:           [H2D] → [Kernel] → [D2H]
Stream 2:           [H2D] → [Kernel] → [D2H]
                     ↓
              并发执行！
```

### 创建和销毁

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
// 使用...
cudaStreamDestroy(stream);
```

### 非阻塞与默认 Stream

```cuda
// 默认 stream (stream 0)
kernel<<<blocks, threads>>>(...);  // 阻塞 host

// 非默认 stream
kernel<<<blocks, threads, 0, stream>>>(...);  // 非阻塞
```

## 并发执行

### 内存传输与 Kernel 并发

```cuda
cudaStream_t streams[4];
for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < 4; i++) {
    int offset = i * chunk_size;

    // 异步 H2D
    cudaMemcpyAsync(d_data[i], h_data + offset,
                    chunk_size, cudaMemcpyHostToDevice, streams[i]);

    // Kernel 执行
    kernel<<<blocks, threads, 0, streams[i]>>>(d_data[i]);

    // 异步 D2H
    cudaMemcpyAsync(h_result + offset, d_data[i],
                    chunk_size, cudaMemcpyDeviceToHost, streams[i]);
}

// 等待所有 stream 完成
for (int i = 0; i < 4; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

### 并发条件

1. **不同 stream**
2. **足够的 GPU 资源**
3. **异步 API** (`cudaMemcpyAsync`)

## CUDA Events

### 创建和销毁

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// 使用...
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 精确计时

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float elapsed;
cudaEventElapsedTime(&elapsed, start, stop);
printf("Kernel 时间：%.3f ms\n", elapsed);
```

### Stream 同步

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// stream1 执行一些操作
kernel1<<<blocks, threads, 0, stream1>>>(...);
cudaEventRecord(event, stream1);

// stream2 等待 stream1 完成
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<blocks, threads, 0, stream2>>>(...);

cudaEventDestroy(event);
```

## 异步内存传输

### cudaMemcpyAsync

```cuda
// 需要 pinned memory
float* h_data;
cudaMallocHost(&h_data, size);  // pinned memory

cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_data, h_data, size,
                cudaMemcpyHostToDevice, stream);

cudaStreamDestroy(stream);
cudaFreeHost(h_data);
```

### 重叠传输和计算

```cuda
__global__ void process(float* data);

cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Stream 1: 传输 + 计算
cudaMemcpyAsync(d_data1, h_data1, size, cudaMemcpyHostToDevice, s1);
process<<<blocks, threads, 0, s1>>>(d_data1);

// Stream 2: 传输 + 计算
cudaMemcpyAsync(d_data2, h_data2, size, cudaMemcpyHostToDevice, s2);
process<<<blocks, threads, 0, s2>>>(d_data2);
```

## 实践：多 Stream 向量运算

### 完整代码

```cuda
#define NUM_STREAMS 4
#define CHUNK_SIZE 1000000

void multi_stream_add(float** h_inputs, float* h_output, int n) {
    cudaStream_t streams[NUM_STREAMS];
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    int chunk = n / NUM_STREAMS;

    // 创建 streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_a[i], chunk * sizeof(float));
        cudaMalloc(&d_b[i], chunk * sizeof(float));
        cudaMalloc(&d_c[i], chunk * sizeof(float));
    }

    // 配置 kernel
    int threads = 256;
    int blocks = (chunk + threads - 1) / threads;

    // 启动所有 streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        // H2D
        cudaMemcpyAsync(d_a[i], h_inputs[0] + i * chunk,
                       chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_inputs[1] + i * chunk,
                       chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Kernel
        vector_add<<<blocks, threads, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], chunk);

        // D2H
        cudaMemcpyAsync(h_output + i * chunk, d_c[i],
                       chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // 等待并清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaFree(d_a[i]); cudaFree(d_b[i]); cudaFree(d_c[i]);
        cudaStreamDestroy(streams[i]);
    }
}
```

## Stream 回调

```cuda
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data) {
    printf("Stream 完成！\n");
}

// 注册回调
cudaStreamAddCallback(stream, my_callback, nullptr, 0);
```

## 性能优化

### 1. 使用多个 Stream

```cuda
// 4 个 streams 并发
for (int i = 0; i < 4; i++) {
    process_stream(streams[i], chunk_i);
}
// 理论上可达 4x 加速
```

### 2. 调整 Chunk 大小

```cuda
// 太小：无法充分利用 GPU
// 太大：并发度降低
int optimal_chunk = total_size / 4;  // 通常 4-8 个 chunks
```

### 3. 使用 cudaStreamNonBlocking

```cuda
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
// 不与 default stream 同步，更好性能
```

## 练习

### TODO: 完成 streams.cu

1. 创建 4 个 streams
2. 实现并发向量加法
3. 使用 events 计时

### 进阶练习

1. 比较单 stream 和多 stream 性能
2. 实现 H2D + Kernel + D2H 重叠
3. 使用 event 进行 stream 间同步

## 调试工具

### 查看 Stream 活动

```bash
nsys profile --stats=true ./your_program
```

### 可视化时间线

```bash
nsys profile -o trace ./your_program
# 生成 trace.qdrep，用 Nsight Systems 打开
```

## 常见问题

### Q1: 什么时候用多个 streams？

- 数据传输与计算重叠
- 多个独立任务并发
- 需要 fine-grained 调度

### Q2: Stream 数量限制？

理论上无限制，但实际 4-16 个通常足够。

### Q3: 为什么我的 stream 不并发？

检查：
1. 是否使用了异步 API
2. GPU 是否有足够资源
3. 是否有不必要的同步

## 下一步

完成 [streams.cu](../tutorials/05_streams/streams.cu) 的 TODO 部分，然后学习 [Lesson 06: CUDA Graph](06_cuda_graph.md)。
