cmake_minimum_required(VERSION 3.14)
project(penguin_walk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find SFML - requires Graphics, Window, and System components.
# Install: apt install libsfml-dev  /  brew install sfml  /  vcpkg install sfml
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# All source is header-only except main.cpp; headers are in src/.

# Original walk cycle viewer
add_executable(penguin_walk_interactive src/main.cpp)
target_include_directories(penguin_walk_interactive PRIVATE src)
target_link_libraries(penguin_walk_interactive PRIVATE
    sfml-graphics
    sfml-window
    sfml-system)

# Interactive WASD-controlled viewer
add_executable(penguin_walk src/main_1.cpp)
target_include_directories(penguin_walk PRIVATE src)
target_link_libraries(penguin_walk PRIVATE
    sfml-graphics
    sfml-window
    sfml-system)

# Useful warnings - we want the compiler to catch any sloppy matrix indexing.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(penguin_walk PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    target_compile_options(penguin_walk_interactive PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()