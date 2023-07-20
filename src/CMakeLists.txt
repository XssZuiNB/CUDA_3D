enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Eigen3 REQUIRED)

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

file(
    GLOB CAMERA_SRC
    "camera/*.cpp"
    "camera/*.hpp"
)

file(
    GLOB CUDA_CONTAINER_SRC
    "cuda_container/*.hpp"
    "cuda_container/*.cu"
    "cuda_container/*.cuh"
)

file(
    GLOB GEOMETRY_SRC
    "geometry/*.cpp"
    "geometry/*.hpp"
    "geometry/*.cu"
    "geometry/*.cuh"
)

file(
    GLOB MOVEMENT_DETECTION_SRC
    "movement_detection/*.cpp"
    "movement_detection/*.hpp"
)

file(
    GLOB UTIL_SRC
    "util/*.cpp"
    "util/*.hpp"    
    "util/*.cu"
    "util/*.cuh"
)

add_library(
    gca_lib 
    SHARED
    ${CAMERA_SRC}
    ${CUDA_CONTAINER_SRC}
    ${GEOMETRY_SRC}
    ${MOVEMENT_DETECTION_SRC}
    ${UTIL_SRC}
)

target_compile_features(
    gca_lib
    PRIVATE
    cxx_std_14
)
    
target_include_directories(
    gca_lib
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
    
target_link_libraries(
    gca_lib
    ${REALSENSE2_FOUND}
    ${OpenCV_LIBS}
    # Eigen3::Eigen
)

set_target_properties(
    gca_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(${CMAKE_VERSION} GREATER_EQUAL 3.18)
    message("CMake version greater than 3.18, set CUDA_ARCHITECTURES as properties")
    set_target_properties(
        gca_lib
        PROPERTIES
        CUDA_ARCHITECTURES "61"
    )

else()
    message("CMake version less than 3.18, set CUDA_ARCHITECTURES in old way")
    set_property(
        TARGET gca_lib 
        PROPERTY CUDA_ARCHITECTURES 61
    )
endif()
