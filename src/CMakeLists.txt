enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

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
    "movement_detection/*.cu"
    "movement_detection/*.cuh"
)

file(
    GLOB REGISTRATION_SRC
    "registration/*.cpp"
    "registration/*.hpp"
    "registration/*.cu"
    "registration/*.cuh"
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
    ${REGISTRATION_SRC}
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
    ${CUDA_LIBRARIES}
    ${REALSENSE2_FOUND}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
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
        CUDA_ARCHITECTURES "61;89"
    )

else()
    message("CMake version less than 3.18, set CUDA_ARCHITECTURES in old way")
    set_property(
        TARGET gca_lib 
        PROPERTY CUDA_ARCHITECTURES "61"
    )
endif()
