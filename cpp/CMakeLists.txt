cmake_minimum_required(VERSION 3.20)
project(RunoCpp LANGUAGES CXX)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui videoio)
add_executable(Runo src/main.cpp)
target_compile_features(Runo PRIVATE cxx_std_17)
target_link_libraries(Runo PRIVATE ${OpenCV_LIBS})

# Preserve the executable location used by Visual Studio.
set_target_properties(Runo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
