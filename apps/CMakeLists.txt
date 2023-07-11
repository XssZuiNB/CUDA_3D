set(app ${CMAKE_PROJECT_NAME})
find_package(PCL REQUIRED)

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
    ../src
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(
    ${app}
    PUBLIC
    gca_lib
    ${PCL_LIBRARIES}
)
