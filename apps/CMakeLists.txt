set(app ${CMAKE_PROJECT_NAME})
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)

add_executable(
    ${app}
    app.cpp
)

target_compile_features(
    ${app}
    PRIVATE
    cxx_std_14
)

target_include_directories(
    ${app}
    PUBLIC
    ../include
    ../src/cuda
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
    ${app}
    PUBLIC
    device_lib
    cuda_lib
    ${PCL_LIBRARIES}
)

if(${CMAKE_VERSION} GREATER_EQUAL 3.18)
    message("CMake version greater than 3.18, set CUDA_ARCHITECTURES as properties")
    set_target_properties(
        ${app}
        PROPERTIES
        CUDA_ARCHITECTURES "52,70"
    )
else()
    message("CMake version less than 3.18, set CUDA_ARCHITECTURES in compile options")
    target_compile_options(${app} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
endif()