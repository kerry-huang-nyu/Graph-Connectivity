cmake_minimum_required(VERSION 3.16)
project(online_graph_connectivity_pybind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11 (provides headers + CMake targets)
find_package(pybind11 CONFIG REQUIRED)

# Find Python interpreter + development headers (Python.h, libpython)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Core C++ library (NO Python / pybind includes here)
add_library(dc STATIC
  dynamic_forest.cpp
  graph.cpp
  hash.cpp
  sequence.cpp
)
target_include_directories(dc PUBLIC ${CMAKE_SOURCE_DIR})

# Python extension module (this file includes pybind11/Python.h)
pybind11_add_module(dynamic_connectivity dynamic_connectivity.cpp)

# Link core logic + Python to the module
target_link_libraries(dynamic_connectivity
  PRIVATE
    dc
    Python::Python
)