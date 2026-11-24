
cmake_minimum_required(VERSION 3.20)

project(transformer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(transformer
    src/main.cpp
    src/core/matrix.cpp
    src/embedding/embedding.cpp
)

target_include_directories(transformer
    PRIVATE
    include
)
