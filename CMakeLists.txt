cmake_minimum_required(VERSION 3.20)

project(
    lattice
    VERSION 0.1.0
    LANGUAGES CXX
)

set(
    CMAKE_CXX_STANDARD 17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED ON
)

set(
    CMAKE_CXX_EXTENSIONS OFF
)

add_library(
    lattice_core
    STATIC
    src/storage.cpp
    src/tensor.cpp
)

target_include_directories(
    lattice_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(
    test_cpp_tensor
    tests/test_cpp_tensor.cpp
)

target_link_libraries(
    test_cpp_tensor
    PRIVATE
    lattice_core
)