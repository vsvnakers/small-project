# Lesson 03: 共享内存
add_executable(shared_mem shared_mem.cu)
set_target_properties(shared_mem PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(shared_mem_solution solution/shared_mem.cu)
set_target_properties(shared_mem_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
