# Lesson 07: Tensor Core
add_executable(tensor_core tensor_core.cu)
set_target_properties(tensor_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# Solution
add_executable(tensor_core_solution solution/tensor_core.cu)
set_target_properties(tensor_core_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
