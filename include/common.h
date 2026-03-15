#ifndef CUDA_TUTORIAL_COMMON_H
#define CUDA_TUTORIAL_COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 错误检查宏
// ============================================================================

// 检查 CUDA API 调用错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 检查 CUDA kernel 启动错误
// 注意：某些 CUDA 版本会输出 PTX toolchain 警告，这不影响功能
// CUDA 12.x 中错误码 223 = "the provided PTX was compiled with an unsupported toolchain"
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        /* 忽略 "PTX compiled with unsupported toolchain" 警告 (err=223) */ \
        if (err != cudaSuccess && err != (cudaError_t)223) { \
            fprintf(stderr, "CUDA kernel error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        /* 同样忽略 PTX toolchain 警告 */ \
        if (err != cudaSuccess && err != (cudaError_t)223) { \
            fprintf(stderr, "CUDA sync error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 检查 cuBLAS 调用错误
#ifdef CUBLAS_CHECK
#undef CUBLAS_CHECK
#endif

// ============================================================================
// 工具函数
// ============================================================================

// 打印设备信息
inline void printDeviceinfo() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== GPU 设备信息 ===\n");
    printf("设备名称：%s\n", prop.name);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("全局内存：%.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("共享内存/块：%lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("每块最大线程数：%d\n", prop.maxThreadsPerBlock);
    printf("多处理器数量：%d\n", prop.multiProcessorCount);
    printf("时钟频率：%.2f GHz\n", prop.clockRate * 1e-6);
    printf(" Warp 大小：%d\n", prop.warpSize);
    printf("==================\n\n");
}

// 格式化字节大小
inline const char* formatBytes(size_t bytes) {
    static char buffer[64];
    if (bytes >= 1024 * 1024 * 1024) {
        sprintf(buffer, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024 * 1024) {
        sprintf(buffer, "%.2f MB", bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        sprintf(buffer, "%.2f KB", bytes / 1024.0);
    } else {
        sprintf(buffer, "%zu B", bytes);
    }
    return buffer;
}

// 验证结果
template<typename T>
inline bool verifyResults(const T* expected, const T* actual, int size,
                          const char* testName, T tolerance = T(1e-5)) {
    for (int i = 0; i < size; i++) {
        T diff = expected[i] - actual[i];
        if (diff < 0) diff = -diff;
        if (diff > tolerance) {
            printf("[错误] %s: 第 %d 个元素不匹配 (期望：%f, 实际：%f)\n",
                   testName, i, (float)expected[i], (float)actual[i]);
            return false;
        }
    }
    printf("[正确] %s: 所有 %d 个元素验证通过\n", testName, size);
    return true;
}

// 简单计时器
class SimpleTimer {
public:
    cudaEvent_t start, stop;

    SimpleTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~SimpleTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startRecord() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    void stopRecord() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
    }

    float elapsedMs() {
        float ms;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

#endif // CUDA_TUTORIAL_COMMON_H
