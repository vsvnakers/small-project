# Lesson 04: 原子操作

## 学习目标

- 理解原子操作的必要性
- 掌握 CUDA 原子操作 API
- 学习实现原子计数器和直方图
- 了解原子操作的性能影响

## 为什么需要原子操作？

### 竞态条件

```cuda
// 错误！多个线程同时写入
__global__ void wrong(float* counter) {
    *counter = *counter + 1;  // 数据竞争！
}

// 执行流程:
// Thread 0: 读取 counter (0)
// Thread 1: 读取 counter (0)  ← 同时！
// Thread 0: 写入 counter (1)
// Thread 1: 写入 counter (1)  ← 应该是 2！
```

### 原子操作解决

```cuda
// 正确！使用原子操作
__global__ void correct(float* counter) {
    atomicAdd(counter, 1);  // 原子地加 1
}
```

## CUDA 原子操作 API

### 支持的类型和操作

| 函数 | 操作 | 类型 |
|------|------|------|
| `atomicAdd` | `*address += val` | int, float, double |
| `atomicSub` | `*address -= val` | int |
| `atomicMul` | `*address *= val` | int |
| `atomicExch` | `*address = val` | int |
| `atomicMax` | `*address = max(*address, val)` | int |
| `atomicMin` | `*address = min(*address, val)` | int |
| `atomicAnd` | `*address &= val` | int |
| `atomicOr` | `*address |= val` | int |
| `atomicXor` | `*address ^= val` | int |
| `atomicCAS` | `*address = (*address == cmp ? val : *address)` | int |

### 浮点数原子操作

```cuda
// CUDA 支持 float 和 double 的 atomicAdd
float* d_counter;
atomicAdd(d_counter, 1.5f);  // ✓
```

## 实战：原子计数器

### 实现

```cuda
__global__ void atomic_counter(int* counter, int n) {
    for (int i = 0; i < n; i++) {
        atomicAdd(counter, 1);
    }
}

int main() {
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // 1000 个线程，每个加 100 次
    atomic_counter<<<100, 1000>>>(d_counter, 100);

    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("结果：%d (期望：%d)\n", result, 100 * 1000);

    cudaFree(d_counter);
}
```

## 实战：直方图

### 基础版本

```cuda
__global__ void histogram_basic(unsigned char* input, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned char value = input[idx];
        atomicAdd(&hist[value], 1);
    }
}
```

### 优化版本（使用共享内存）

```cuda
__global__ void histogram_optimized(unsigned char* input, int* hist, int n) {
    __shared__ int local_hist[256];
    int tid = threadIdx.x;

    // 初始化共享内存
    for (int i = tid; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // 在共享内存中累积
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        atomicAdd(&local_hist[input[idx]], 1);
    }
    __syncthreads();

    // 合并到全局内存
    for (int i = tid; i < 256; i += blockDim.x) {
        atomicAdd(&hist[i], local_hist[i]);
    }
}
```

### 性能对比

| 版本 | 时间 | 加速 |
|------|------|------|
| 全局原子 | 2.5 ms | 1x |
| 共享内存 | 0.8 ms | 3.1x |

## 原子操作性能

### 延迟对比

| 操作 | 延迟 |
|------|------|
| 普通加法 | ~1 cycle |
| atomicAdd (L2) | ~500 cycles |
| atomicAdd (全局) | ~1000 cycles |

### 减少原子操作

```cuda
// 方法 1: 私有累加
float sum = 0;
for (int i = 0; i < chunk_size; i++) {
    sum += data[i];
}
atomicAdd(&total, sum);  // 只原子操作一次

// 方法 2: 使用共享内存
__shared__ float sdata[256];
// 在 block 内归约
if (tid == 0) {
    atomicAdd(&total, sdata[0]);
}
```

## 原子比较交换（CAS）

### 实现自旋锁

```cuda
__global__ void spinlock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // 等待锁
    }

    // 临界区
    ...

    // 释放锁
    atomicExch(lock, 0);
}
```

### 实现链表插入

```cuda
__global__ void list_insert(Node** head, Node* new_node) {
    Node* old_head;
    do {
        old_head = *head;
        new_node->next = old_head;
    } while (atomicCAS((int**)head, (int*)old_head, (int*)new_node) != (int*)old_head);
}
```

## 练习

### TODO: 完成 atomic_ops.cu

1. 实现原子计数器
2. 实现原子直方图
3. 验证结果正确性

### 进阶练习

1. 实现共享内存优化的直方图
2. 比较两种实现的性能
3. 尝试实现自旋锁

## 调试原子操作

### 验证正确性

```cuda
// 期望结果应该等于线程数 × 每个线程的操作数
int expected = num_blocks * threads_per_block * operations_per_thread;
printf("实际：%d, 期望：%d\n", result, expected);
```

### 检测数据竞争

```bash
cuda-memcheck --tool racecheck ./your_program
```

## 常见问题

### Q1: 原子操作一定慢吗？

不一定。在以下情况下性能可以接受：

1. 冲突不频繁
2. 使用共享内存减少冲突
3. Ampere 架构有更好的原子操作性能

### Q2: 双精度原子操作支持吗？

- 计算能力 6.0+ 支持 `atomicAdd(double*, double)`
- 旧架构需要自己实现

### Q3: 如何最大化原子操作性能？

```cuda
// 1. 尽量减少全局原子操作
// 2. 使用共享内存作为 buffer
// 3. 在 Ampere GPU 上利用新的原子操作特性
```

## 下一步

完成 [atomic_ops.cu](../tutorials/04_atomic_ops/atomic_ops.cu) 的 TODO 部分，然后学习 [Lesson 05: Streams & Events](05_streams_events.md)。
