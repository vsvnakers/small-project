# CUDA 教程文档索引

本目录包含完整的 CUDA 教程，从基础到高级主题。

## 课程列表

| 课号 | 主题 | 文档 | 代码目录 |
|------|------|------|----------|
| 00 | CUDA 基础与环境设置 | [00_cuda_basics.md](docs/00_cuda_basics.md) | `tutorials/00_basics/` |
| 01 | 内存模型 | [01_memory_model.md](docs/01_memory_model.md) | `tutorials/01_vector_add/` |
| 02 | 线程层次结构 | [02_thread_hierarchy.md](docs/02_thread_hierarchy.md) | `tutorials/02_matrix_mul/` |
| 03 | 共享内存 | [03_shared_memory.md](docs/03_shared_memory.md) | `tutorials/03_shared_memory/` |
| 04 | 原子操作 | [04_atomic_operations.md](docs/04_atomic_operations.md) | `tutorials/04_atomic_ops/` |
| 05 | Streams & Events | [05_streams_events.md](docs/05_streams_events.md) | `tutorials/05_streams/` |
| 06 | CUDA Graph | [06_cuda_graph.md](docs/06_cuda_graph.md) | `tutorials/06_cuda_graph/` |
| 07 | Tensor Core | [07_tensor_core.md](docs/07_tensor_core.md) | `tutorials/07_tensor_core/` |
| 08 | cuBLAS 库 | [08_cublas.md](docs/08_cublas.md) | `tutorials/08_cublas/` |
| 09 | 性能优化 | [09_performance_optimization.md](docs/09_performance_optimization.md) | `tutorials/09_optimization/` |

## 学习路径

```
基础篇 (Lesson 00-02)
├── GPU 架构简介
├── CUDA 编程模型
├── 内存管理
└── 线程索引计算

进阶篇 (Lesson 03-05)
├── 共享内存优化
├── 原子操作与同步
└── 并发执行 (Streams/Events)

高级篇 (Lesson 06-09)
├── CUDA Graph
├── Tensor Core 编程
├── cuBLAS 库使用
└── 性能分析与优化
```

## 编译说明

### 环境要求

- CUDA Toolkit 11.0 或更高版本
- CMake 3.18 或更高版本
- 支持 CUDA 的 GPU（建议 RTX 30/40 系列）

### 编译步骤

```bash
# 创建 build 目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译所有教程
make -j$(nproc)

# 或者编译单个教程
make vector_add_solution
```

### 运行测试

```bash
# 进入对应教程目录
cd build/tutorials/01_vector_add/

# 运行参考答案
./vector_add_solution
```

## 代码结构说明

每个教程包含以下文件：

- `xxx.cu` - TODO 版本（需要你完成）
- `solution/xxx.cu` - 参考答案
- `CMakeLists.txt` - 编译配置

## 学习建议

1. **先阅读文档**：理解概念后再动手写代码
2. **完成 TODO**：先尝试自己实现，再查看答案
3. **运行验证**：确保你的代码输出正确结果
4. **性能分析**：使用 Nsight 工具分析性能

## 相关资源

- [NVIDIA CUDA 文档](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
