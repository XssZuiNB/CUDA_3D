cmake_minimum_required(VERSION 3.14)

project(Collision_Avoidance LANGUAGES CXX CUDA)

find_package(realsense2 REQUIRED)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(src/cuda)

add_executable(${CMAKE_PROJECT_NAME} apps/app.cpp)