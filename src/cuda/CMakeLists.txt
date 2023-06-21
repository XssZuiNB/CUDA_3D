enable_language(CUDA)

find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)

add_library(cuda_lib
    cuda_point_cloud.cu
)

set_target_properties(
    cuda_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(
    cuda_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../include
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(${CMAKE_VERSION} GREATER_EQUAL 3.18)
    message("CMake version greater than 3.18, set CUDA_ARCHITECTURES as properties")
    set_target_properties(
        cuda_lib
        PROPERTIES
        CUDA_ARCHITECTURES "70"
    )
else()
    message("CMake version less than 3.18, set CUDA_ARCHITECTURES in compile options")
    target_compile_options(cuda_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_89,code=sm_89>)
endif()

target_compile_definitions(
    cuda_lib
    PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(cuda_lib
    PUBLIC
    ${PCL_LIBRARIES}
)
