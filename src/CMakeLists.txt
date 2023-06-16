cmake_minimum_required(VERSION 3.14)

add_subdirectory(cuda)

find_package(realsense2 REQUIRED)

add_library(
    gca_lib 
    device.cpp
    )

target_compile_features(
    gca_lib 
    PRIVATE 
    cxx_std_14
    )

target_include_directories(
    gca_lib 
    PUBLIC 
    ../include
    )
