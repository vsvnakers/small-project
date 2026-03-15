# Lesson 05: Streams 和 Events
add_executable(streams streams.cu)
set_target_properties(streams PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Solution
add_executable(streams_solution solution/streams.cu)
set_target_properties(streams_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
