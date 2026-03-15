# Lesson 00: CUDA 基础
add_executable(hello_world hello_world.cu)
set_target_properties(hello_world PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(hello_world_solution solution/hello_world.cu)
set_target_properties(hello_world_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
