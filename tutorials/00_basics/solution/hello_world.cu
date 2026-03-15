/**
 * Lesson 00: CUDA 基础 - Hello World (参考答案)
 *
 * 编译：nvcc -o hello_world_solution hello_world.cu
 * 运行：./hello_world_solution
 */

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel 函数 - 在 GPU 上执行
// 注意：kernel 内部的 printf 在某些系统 (如 WSL2) 上可能无法显示
// 这不影响 CUDA 功能，只是显示问题
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

    // 注意：如果看不到 "Hello from thread X" 输出，这是正常的
    // 某些系统 (WSL2/特定驱动) 会缓冲 CUDA kernel 的 printf 输出
    // 这不影响 CUDA 功能，可以查看设备信息验证 GPU 正常工作
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU 设备：%s\n", prop.name);
    printf("(如果看不到线程输出，这是正常的 - CUDA printf 在某些系统上被缓冲)\n");

    printf("\nCUDA 程序完成!\n");

    return 0;
}
