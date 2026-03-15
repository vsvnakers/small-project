# Lesson 01: 向量加法
add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(vector_add PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Solution
add_executable(vector_add_solution solution/vector_add.cu)
set_target_properties(vector_add_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(vector_add_solution PRIVATE ${CMAKE_SOURCE_DIR}/include)
