# CUDA 学习教程

一个完整的 CUDA 编程教程，从基础到高级主题，包含详细的文档、TODO 练习和参考答案。

## 📚 课程大纲

| 课号 | 主题 | 难度 | 前置要求 |
|------|------|------|----------|
| [00](docs/00_cuda_basics.md) | CUDA 基础与环境设置 | ⭐ | 无 |
| [01](docs/01_memory_model.md) | 内存模型 | ⭐ | Lesson 00 |
| [02](docs/02_thread_hierarchy.md) | 线程层次结构 | ⭐⭐ | Lesson 01 |
| [03](docs/03_shared_memory.md) | 共享内存 | ⭐⭐ | Lesson 02 |
| [04](docs/04_atomic_operations.md) | 原子操作 | ⭐⭐ | Lesson 03 |
| [05](docs/05_streams_events.md) | Streams & Events | ⭐⭐⭐ | Lesson 04 |
| [06](docs/06_cuda_graph.md) | CUDA Graph | ⭐⭐⭐ | Lesson 05 |
| [07](docs/07_tensor_core.md) | Tensor Core | ⭐⭐⭐⭐ | Lesson 06 |
| [08](docs/08_cublas.md) | cuBLAS 库 | ⭐⭐⭐ | Lesson 07 |
| [09](docs/09_performance_optimization.md) | 性能优化 | ⭐⭐⭐⭐ | Lesson 08 |

## 🚀 快速开始

### 环境要求

- **操作系统**: Linux / Windows / WSL2
- **CUDA Toolkit**: 11.0 或更高版本
- **CMake**: 3.18 或更高版本
- **GPU**: 支持 CUDA 的 NVIDIA GPU（建议 RTX 30/40 系列）

### 检查环境

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 信息
nvidia-smi
```

### 编译项目

```bash
# 克隆仓库
git clone <repo-url>
cd small-project

# 切换到 CUDA 教程分支
git checkout cuda-tutorial

# 创建 build 目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译所有教程
make -j$(nproc)
```

### 运行教程

```bash
# 进入教程目录
cd build/tutorials/01_vector_add/

# 运行参考答案
./vector_add_solution

# 或者先完成 TODO 版本，然后运行
./vector_add
```

## 📁 项目结构

```
cuda_project/
├── README.md                    # 本文件
├── CMakeLists.txt               # 主构建配置
├── docs/                        # 教程文档
│   ├── 00_cuda_basics.md
│   ├── 01_memory_model.md
│   ├── ...
│   └── 09_performance_optimization.md
├── tutorials/                   # 每课的代码
│   ├── 00_basics/
│   │   ├── hello_world.cu       # TODO 版本
│   │   ├── solution/
│   │   │   └── hello_world.cu   # 参考答案
│   │   └── CMakeLists.txt
│   ├── 01_vector_add/
│   ├── ...
│   └── 09_optimization/
├── include/
│   └── common.h                 # 公共工具函数
└── scripts/
    └── check_gpu.sh             # GPU 检查脚本
```

## 📖 如何使用

### 学习流程

1. **阅读文档**: 先阅读 `docs/` 中对应课程的教程
2. **理解概念**: 理解 GPU 架构和 CUDA 编程模型
3. **完成 TODO**: 打开 `tutorials/XX_*/xxx.cu` 完成练习
4. **运行验证**: 编译并运行你的代码
5. **对比答案**: 与 `solution/` 中的参考答案对比
6. **性能分析**: 使用 Nsight 工具分析性能

### 代码约定

每个教程包含两种版本：

- **TODO 版本** (`xxx.cu`): 需要你完成的代码
- **Solution 版本** (`solution/xxx.cu`): 完整的参考答案

```cuda
/**
 * Lesson XX: 课程名称
 *
 * TODO:
 * 1. 任务 1
 * 2. 任务 2
 */

// 你的代码写在这里
__global__ void kernel(...) {
    // TODO: 实现...
}
```

## 🛠️ 工具

### 编译工具

| 工具 | 用途 |
|------|------|
| nvcc | CUDA 编译器 |
| cmake | 构建系统生成器 |
| make | 编译执行 |

### 调试工具

| 工具 | 用途 |
|------|------|
| cuda-gdb | CUDA 调试器 |
| cuda-memcheck | 内存检查 |
| Nsight Systems | 系统级 profiling |
| Nsight Compute | Kernel 级 profiling |

### 使用示例

```bash
# 调试
cuda-gdb ./your_program

# 内存检查
cuda-memcheck ./your_program

# Profiling
nsys profile --stats=true ./your_program
ncu ./your_program
```

## 📋 学习检查清单

### 基础篇 □

- [ ] 理解 GPU 和 CPU 的区别
- [ ] 掌握 CUDA 内存管理
- [ ] 会计算线程索引
- [ ] 能编写基本 kernel

### 进阶篇 □

- [ ] 使用共享内存优化
- [ ] 实现原子操作
- [ ] 使用 Streams 并发
- [ ] 使用 Events 计时

### 高级篇 □

- [ ] 理解 CUDA Graph
- [ ] 使用 Tensor Core
- [ ] 调用 cuBLAS 库
- [ ] 进行性能优化

## 🔧 常见问题

### Q: 编译失败怎么办？

```bash
# 检查 CUDA 安装
which nvcc
nvcc --version

# 检查 CUDA_HOME
echo $CUDA_HOME

# 重新配置 CMake
rm -rf build
mkdir build && cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

### Q: 运行时错误？

```bash
# 检查 GPU 是否可见
nvidia-smi

# 检查 CUDA 设备
cudaDeviceQuery
```

### Q: 性能不如预期？

1. 使用 Nsight Compute 分析瓶颈
2. 检查内存访问模式
3. 检查 occupancy
4. 确认 Tensor Core 是否启用

## 📚 更多资源

### 官方文档

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

### 推荐书籍

- 《CUDA C 编程权威指南》
- 《Programming Massively Parallel Processors》

### 在线课程

- [Udacity: Intro to Parallel Programming](https://www.udacity.com/course/cs344)
- [Coursera: Heterogeneous Parallel Programming](https://www.coursera.org/learn/heterogeneous-parallel-programming)

## 📝 许可证

本教程用于学习目的，可自由使用和修改。

## 🙏 贡献

欢迎提交 Issue 和 Pull Request 改进教程内容。

---

**祝你学习愉快！** 🚀

如有问题，请在 GitHub 上提 Issue。
