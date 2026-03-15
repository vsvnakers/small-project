/**
 * Lesson 00: CUDA 基础 - Hello World
 *
 * 目标：了解 CUDA 程序的基本结构
 *
 * TODO:
 * 1. 补全 kernel 函数，让每个线程打印自己的线程 ID
 * 2. 理解 <<<grid, block>>> 的含义
 * 3. 观察输出结果，理解线程的执行顺序
 */

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel 函数 - 在 GPU 上执行
// TODO: 实现这个函数，让每个线程打印自己的全局线程 ID
__global__ void hello_kernel() {
    // 提示：使用 threadIdx 和 blockIdx 计算全局线程 ID
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello from thread %d\n", idx);
}

int main() {
    // 配置执行参数
    int threads_per_block = 4;    // 每个 block 有 4 个线程
    int number_of_blocks = 4;     // 4 个 blocks

    printf("启动 CUDA kernel...\n");
    printf("配置：%d blocks x %d threads = %d 线程\n",
           number_of_blocks, threads_per_block,
           number_of_blocks * threads_per_block);

    // 启动 kernel
    // TODO: 补全 kernel 调用参数
    hello_kernel<<<number_of_blocks, threads_per_block>>>();

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    printf("CUDA 程序完成!\n");

    return 0;
}
