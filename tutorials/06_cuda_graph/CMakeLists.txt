# Lesson 06: CUDA Graph
add_executable(cuda_graph cuda_graph.cu)
set_target_properties(cuda_graph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(cuda_graph_solution solution/cuda_graph.cu)
set_target_properties(cuda_graph_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
