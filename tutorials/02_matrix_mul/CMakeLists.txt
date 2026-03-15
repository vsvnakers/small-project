# Lesson 02: 矩阵乘法
add_executable(matrix_mul matrix_mul.cu)
set_target_properties(matrix_mul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(matrix_mul_solution solution/matrix_mul.cu)
set_target_properties(matrix_mul_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
