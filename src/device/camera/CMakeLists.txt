cmake_minimum_required(VERSION  3.12)
project(camera)
add_library(${PROJECT_NAME}
    STATIC
    camera.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    spdlog::spdlog
)


# 下面这段是我复制的
message("Link OpenCV build library via OpenCVConfig.cmake")
add_definitions(-D LINK_OPENCV_LIB_AUTO)
find_package(OpenCV REQUIRED)
include_directories(camera ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera ${OpenCV_LIBS})
