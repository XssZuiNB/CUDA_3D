cmake_minimum_required(VERSION 3.14)

project(Collision_Avoidance LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")

include_directories(src)

add_subdirectory(src)
add_subdirectory(apps)
