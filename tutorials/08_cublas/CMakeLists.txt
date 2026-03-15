# Lesson 08: cuBLAS
add_executable(cublas_example cublas_example.cu)
set_target_properties(cublas_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cublas_example PRIVATE ${CUBLAS_LIBRARIES})
target_include_directories(cublas_example PRIVATE ${CUBLAS_INCLUDE_DIR})

# Solution
add_executable(cublas_example_solution solution/cublas_example.cu)
set_target_properties(cublas_example_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cublas_example_solution PRIVATE ${CUBLAS_LIBRARIES})
target_include_directories(cublas_example_solution PRIVATE ${CUBLAS_INCLUDE_DIR})
