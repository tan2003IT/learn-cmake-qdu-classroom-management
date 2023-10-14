cmake_minimum_required(VERSION 3.12)
project(exe_classroom)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    camera
    sets
    computer
    course
    self_learn
)

add_dependencies(${PROJECT_NAME}
    camera
    sets
    computer
    course
    self_learn
)