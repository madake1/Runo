cmake_minimum_required(VERSION 3.20)

project(Runo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)

add_executable(Runo test.cpp)

target_link_libraries(Runo PRIVATE ${OpenCV_LIBS})