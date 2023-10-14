cmake_minimum_required(VERSION 3.12)
project(self_learn)

add_library(${PROJECT_NAME}
    STATIC
    self_learn.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    sets
    camera
    computer
)

target_link_libraries(${PROJECT_NAME}
    spdlog::spdlog
    camera
    sets
    computer
)

add_dependencies(${PROJECT_NAME}
    camera
    sets
    computer
)
include_directories()