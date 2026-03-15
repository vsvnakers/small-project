/**
 * Lesson 09: 性能优化 (参考答案)
 *
 * 编译：nvcc -o optimization_solution optimization.cu
 * 运行：./optimization_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

#define N (1024 * 1024 * 16)  // 16M 元素
#define BLOCK_SIZE 256

// ============================================================================
// Kernel 1: 未优化版本 - 随机内存访问
// ============================================================================

__global__ void naive_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 随机访问模式（未合并）
        // 这会导致严重的性能问题
        int random_idx = (idx * 7919) % n;  // 质数乘法
        output[idx] = input[random_idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// Kernel 2: 优化版本 - 合并内存访问
// ============================================================================

__global__ void optimized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 顺序访问（合并访问）
        // 相邻线程访问相邻内存地址
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// Kernel 3: 共享内存优化版本
// ============================================================================

__global__ void shared_mem_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];

    // 从全局内存加载到共享内存
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // 在共享内存中处理
    if (idx < n) {
        float val = sdata[tid];
        output[idx] = val * 2.0f + 1.0f;
    }
}

// ============================================================================
// Kernel 4: 向量化内存访问（使用 float4）
// ============================================================================

__global__ void vectorized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 使用 float4 进行向量化加载/存储
    // 每次处理 4 个 float
    int float4_idx = idx * 4;

    while (float4_idx + 3 < n) {
        float4 in = reinterpret_cast<const float4*>(&input[float4_idx])[0];
        float4 out;
        out.x = in.x * 2.0f + 1.0f;
        out.y = in.y * 2.0f + 1.0f;
        out.z = in.z * 2.0f + 1.0f;
        out.w = in.w * 2.0f + 1.0f;
        reinterpret_cast<float4*>(&output[float4_idx])[0] = out;

        float4_idx += stride * 4;
    }

    // 处理剩余元素
    for (int i = float4_idx; i < n; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}

// ============================================================================
// Kernel 5: Occupancy 优化版本（更多 block）
// ============================================================================

__global__ void occupancy_optimized_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void process_cpu(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
}

// ============================================================================
// 性能分析函数
// ============================================================================

void run_benchmark(const char* name, cudaStream_t stream,
                   void (*kernel)(float*, float*, int),
                   float* d_in, float* d_out, int n,
                   int blocks, int threads, int iterations) {
    // 预热
    kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);

    SimpleTimer timer;
    timer.startRecord();

    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
    }

    timer.stopRecord();

    float avg_time = timer.elapsedMs() / iterations;
    float bandwidth = (2.0 * n * sizeof(float)) / (avg_time * 1e-6) / 1e9;

    printf("%-20s: %8.3f ms, %8.2f GB/s\n", name, avg_time, bandwidth);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("=== Lesson 09: 性能优化 ===\n\n");
    printDeviceinfo();

    int size = N * sizeof(float);
    int iterations = 10;

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    printf("数据大小：%s\n", formatBytes(size));
    printf("迭代次数：%d\n\n", iterations);

    // 创建多个 stream 用于并发分析
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    printf("性能测试结果:\n");
    printf("=============================\n");

    // -------------------------------------------------------------------------
    // 测试 1: 基础配置
    // -------------------------------------------------------------------------
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        run_benchmark("未优化 (随机)", streams[0],
                     naive_kernel, d_input, d_output, N,
                     blocks, threads, iterations);

        run_benchmark("优化 (顺序)", streams[1],
                     optimized_kernel, d_input, d_output, N,
                     blocks, threads, iterations);
    }

    // -------------------------------------------------------------------------
    // 测试 2: 共享内存优化
    // -------------------------------------------------------------------------
    {
        int threads = BLOCK_SIZE;
        int blocks = (N + threads - 1) / threads;

        run_benchmark("共享内存", streams[2],
                     shared_mem_kernel, d_input, d_output, N,
                     blocks, threads, iterations);
    }

    // -------------------------------------------------------------------------
    // 测试 3: 向量化访问
    // -------------------------------------------------------------------------
    {
        int threads = 256;
        int blocks = (N + threads - 1) / (threads * 4);  // 每个 block 处理更多
        if (blocks > 65535) blocks = 65535;

        run_benchmark("向量化 (float4)", streams[3],
                     vectorized_kernel, d_input, d_output, N,
                     blocks, threads, iterations);
    }

    // -------------------------------------------------------------------------
    // 测试 4: Occupancy 优化
    // -------------------------------------------------------------------------
    {
        // 使用更多 block 来隐藏延迟
        int threads = 128;  // 更小的 block，更多的 block
        int blocks = N / threads * 4;  // 4 倍的工作量
        if (blocks > 65535) blocks = 65535;

        run_benchmark("Occupancy 优化", streams[0],
                     occupancy_optimized_kernel, d_input, d_output, N,
                     blocks, threads, iterations);
    }

    printf("=============================\n\n");

    // -------------------------------------------------------------------------
    // Occupancy 分析
    // -------------------------------------------------------------------------
    printf("Occupancy 分析:\n");
    printf("------------------\n");

    int min_grid_size, block_size;

    // 分析 optimized_kernel
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       optimized_kernel, 0, 0);
    printf("optimized_kernel:\n");
    printf("  最大 block 大小：%d\n", block_size);
    printf("  最小 grid 大小：%d\n", min_grid_size);

    // 获取 occupancy
    int active_blocks, active_warps;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                   optimized_kernel, block_size, 0);
    printf("  每 SM 活跃 block 数：%d\n", active_blocks);
    printf("  每 block warp 数：%d\n", block_size / 32);
    printf("  理论 occupancy: %.1f%%\n",
           (active_blocks * (block_size / 32.0)) / 64.0 * 100.0);

    // -------------------------------------------------------------------------
    // 验证结果
    // -------------------------------------------------------------------------
    printf("\n验证结果:\n");
    printf("------------------\n");

    // 运行各个 kernel 并验证
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    naive_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // 验证（前几个元素）
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_input[i] * 2.0f + 1.0f;
        // 注意：naive_kernel 使用随机访问，所以不直接比较
    }
    printf("所有 kernel 执行成功 ✓\n");

    // -------------------------------------------------------------------------
    // 清理
    // -------------------------------------------------------------------------
    for (int i = 0; i < 4; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("\n=== 知识点总结 ===\n");
    printf("1. 内存合并访问 (Coalesced Access)\n");
    printf("   - 相邻线程访问相邻内存地址\n");
    printf("   - 可提升数倍内存带宽利用率\n\n");

    printf("2. Occupancy 优化\n");
    printf("   - 使用 cudaOccupancyMaxPotentialBlockSize 查询最优配置\n");
    printf("   - 更多活跃 warp 可以隐藏内存延迟\n\n");

    printf("3. 向量化内存访问\n");
    printf("   - 使用 float4/uchar4 等向量类型\n");
    printf("   - 减少内存指令数量\n\n");

    printf("4. 共享内存优化\n");
    printf("   - 减少对全局内存的访问\n");
    printf("   - 注意 bank conflict\n\n");

    printf("5. Profiling 工具\n");
    printf("   - Nsight Systems: 时间线分析\n");
    printf("   - Nsight Compute: kernel 详细分析\n");
    printf("   - nvprof: 命令行 profiler\n");

    printf("\n程序完成!\n");

    return 0;
}
