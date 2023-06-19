set(app ${CMAKE_PROJECT_NAME})

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
    )

