find_package(OpenCV REQUIRED)

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)

if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

file(GLOB SOURCE_FILES "*.cpp")

add_library(
    device_lib
    ${SOURCE_FILES}
)

target_compile_features(
    device_lib
    PRIVATE
    cxx_std_14
)

target_include_directories(
    device_lib
    PUBLIC
    ../../include
    ${${OpenCV_INCLUDE_DIRS}}
)

target_link_libraries(
    device_lib
    ${REALSENSE2_FOUND}
    ${OpenCV_LIBS}
)
