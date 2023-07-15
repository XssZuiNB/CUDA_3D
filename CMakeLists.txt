cmake_minimum_required(VERSION 3.14)

project(Collision_Avoidance LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(src)

add_subdirectory(src)
add_subdirectory(apps)
