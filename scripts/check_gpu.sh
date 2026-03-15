#!/bin/bash
# GPU 信息检查脚本

echo "=== GPU 信息检查 ==="
echo ""

# 检查 nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "=== nvidia-smi 输出 ==="
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    echo ""
else
    echo "警告：未找到 nvidia-smi"
    echo ""
fi

# 检查 nvcc
if command -v nvcc &> /dev/null; then
    echo "=== CUDA 版本 ==="
    nvcc --version | head -2
    echo ""
else
    echo "警告：未找到 nvcc"
    echo ""
fi

# 检查 CUDA deviceQuery
if [ -f /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery ]; then
    echo "=== DeviceQuery ==="
    /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery 2>/dev/null | grep -E "(CUDA Capability|Total amount of global memory|Maximum number of threads per block)"
elif command -v cudaDeviceQuery &> /dev/null; then
    cudaDeviceQuery 2>/dev/null | grep -E "(CUDA Capability|Total amount of global memory|Maximum number of threads per block)"
else
    echo "提示：安装 CUDA samples 后可运行 deviceQuery 查看详细信息"
fi

echo ""
echo "=== 检查完成 ==="
