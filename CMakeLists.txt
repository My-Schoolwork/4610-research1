cmake_minimum_required(VERSION 3.14)
project(penguin_walk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# All source is header-only except main.cpp; headers are in src/.
add_executable(penguin_walk src/main.cpp)
target_include_directories(penguin_walk PRIVATE src)

# Useful warnings - we want the compiler to catch any sloppy matrix indexing.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(penguin_walk PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()
