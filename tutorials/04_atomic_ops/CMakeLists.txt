# Lesson 04: 原子操作
add_executable(atomic_ops atomic_ops.cu)
set_target_properties(atomic_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(atomic_ops_solution solution/atomic_ops.cu)
set_target_properties(atomic_ops_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
