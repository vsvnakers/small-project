# Lesson 09: 性能优化
add_executable(optimization optimization.cu)
set_target_properties(optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(optimization_solution solution/optimization.cu)
set_target_properties(optimization_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
