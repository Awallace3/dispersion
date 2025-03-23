cmake_minimum_required(VERSION 3.10)

project(DISPERSION VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(pybind11_DIR pybind11)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(OpenMP REQUIRED)

add_subdirectory(dftd4)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

add_subdirectory(pybind11)

pybind11_add_module(dispersion src/binder.cpp src/disp.cpp src/constants.cpp)
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
