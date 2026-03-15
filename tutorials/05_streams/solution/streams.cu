/**
 * Lesson 05: Streams 和 Events (参考答案)
 *
 * 编译：nvcc -o streams_solution streams.cu
 * 运行：./streams_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N 1000000
#define NUM_STREAMS 4

// ============================================================================
// Kernel 定义
// ============================================================================

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_mul(float* a, float* b, float* c, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] * factor;
    }
}

// ============================================================================
// 多 stream 示例
// ============================================================================

void multi_stream_example(float* h_a, float* h_b, float* h_c, int n) {
    int elements_per_stream = n / NUM_STREAMS;
    int stream_size = elements_per_stream * sizeof(float);

    // 1. 创建多个 streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 2. 为每个 stream 分配 device 内存
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], stream_size));
        CUDA_CHECK(cudaMalloc(&d_b[i], stream_size));
        CUDA_CHECK(cudaMalloc(&d_c[i], stream_size));
    }

    // 3. 配置 kernel
    int threads = 256;
    int blocks = (elements_per_stream + threads - 1) / threads;

    // 4. 在每个 stream 中执行操作
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * elements_per_stream;

        // H2D 拷贝到这个 stream
        CUDA_CHECK(cudaMemcpyAsync(d_a[i], h_a + offset, stream_size,
                                   cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[i], h_b + offset, stream_size,
                                   cudaMemcpyHostToDevice, streams[i]));

        // 启动 kernel（自动在这个 stream 中执行）
        vector_add<<<blocks, threads, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], elements_per_stream);

        // D2H 拷贝
        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c[i], stream_size,
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // 5. 等待所有 streams 完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaFree(d_a[i]));
        CUDA_CHECK(cudaFree(d_b[i]));
        CUDA_CHECK(cudaFree(d_c[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

// ============================================================================
// 不同 kernel 的并发执行
// ============================================================================

void concurrent_kernels(float* d_a, float* d_b, float* d_c, float* d_d, int n) {
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int elements_per_stream = n / NUM_STREAMS;
    int threads = 256;
    int blocks = (elements_per_stream + threads - 1) / threads;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * elements_per_stream;

        // 偶数 stream 执行加法，奇数 stream 执行乘法
        if (i % 2 == 0) {
            vector_add<<<blocks, threads, 0, streams[i]>>>(
                d_a + offset, d_b + offset, d_c + offset, elements_per_stream);
        } else {
            vector_mul<<<blocks, threads, 0, streams[i]>>>(
                d_a + offset, d_b + offset, d_d + offset, elements_per_stream, 2.0f);
        }
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 05: Streams 和 Events ===\n\n");
    printDeviceinfo();

    int size = N * sizeof(float);
    int elements_per_stream = N / NUM_STREAMS;

    // 分配 host 内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 0.5f;
    }

    // 分配 device 内存
    float *d_a, *d_b, *d_c, *d_d;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMalloc(&d_d, size));

    // 预拷贝数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 测试 1: 默认 stream（顺序执行）
    printf("测试 1: 默认 stream (顺序执行)\n");
    printf("--------------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        // 4 次独立的 kernel 调用，顺序执行
        for (int i = 0; i < 4; i++) {
            vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stopRecord();
        printf("时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 测试 2: 多个 streams（并发执行）
    printf("测试 2: 多个 streams (并发执行)\n");
    printf("--------------------------------\n");
    {
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        int threads = 256;
        int blocks = (elements_per_stream + threads - 1) / threads;

        SimpleTimer timer;
        timer.startRecord();

        for (int i = 0; i < NUM_STREAMS; i++) {
            int offset = i * elements_per_stream;
            vector_add<<<blocks, threads, 0, streams[i]>>>(
                d_a + offset, d_b + offset, d_c + offset, elements_per_stream);
        }

        // 等待所有 streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }

        timer.stopRecord();
        printf("时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 测试 3: 并发 kernel（不同操作）
    printf("测试 3: 并发 Kernels (不同操作)\n");
    printf("--------------------------------\n");
    {
        SimpleTimer timer;
        timer.startRecord();

        concurrent_kernels(d_a, d_b, d_c, d_d, N);

        timer.stopRecord();
        printf("时间：%.3f ms\n\n", timer.elapsedMs());
    }

    // 测试 4: Events 精确计时
    printf("测试 4: Events 精确计时\n");
    printf("----------------------\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Kernel 执行时间 (Event): %.3f ms\n", elapsed);

    // 计算 throughput
    float bandwidth = (3 * N * sizeof(float)) / (elapsed * 1e-6) / 1e9;
    printf("内存带宽：%.2f GB/s\n", bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 清理
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n=== 知识点总结 ===\n");
    printf("1. Stream 是 GPU 操作的队列，不同 stream 可并发执行\n");
    printf("2. 使用 cudaMemcpyAsync 进行异步内存传输\n");
    printf("3. Event 用于精确计时和 stream 同步\n");
    printf("4. 并发执行可提升 GPU 利用率\n");

    printf("\n程序完成!\n");

    return 0;
}
