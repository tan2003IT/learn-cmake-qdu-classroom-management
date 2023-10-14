cmake_minimum_required(VERSION 3.12)
project(course)

add_library(${PROJECT_NAME}
    STATIC
    course.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    camera
    computer
    sets
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    camera
    sets
    computer
)

add_dependencies(${PROJECT_NAME}
    camera
    sets
    computer
)