/**
 * Lesson 00: CUDA 基础 - Hello World (参考答案)
 *
 * 编译：nvcc -o hello_world_solution hello_world.cu
 * 运行：./hello_world_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel 函数 - 在 GPU 上执行
__global__ void hello_kernel() {
    // 计算全局线程 ID
    // blockIdx.x: 当前 block 在 grid 中的索引
    // blockDim.x: 每个 block 的线程数
    // threadIdx.x: 当前线程在 block 中的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程打印自己的 ID
    printf("Hello from thread %d\n", idx);
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
    // 语法：kernel_name<<<grid_size, block_size>>>(arguments)
    hello_kernel<<<number_of_blocks, threads_per_block>>>();

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    printf("\nCUDA 程序完成!\n");

    return 0;
}
