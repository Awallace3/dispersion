cmake_minimum_required(VERSION 3.10)

project(DISPERSION VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(pybind11_DIR pybind11)
# find_package(pybind11 REQUIRED)

if(UNIX AND NOT APPLE)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++-12")
    include_directories(/usr/local/include
        /usr/local/include/eigen3
        /usr/include
        /usr/lib
        pybind11/include
        )
    link_directories(/usr/local/lib
        /usr/local/include/eigen3
        /usr/include
        /usr/lib
        pybind11/include
        )
endif(UNIX AND NOT APPLE)
if(UNIX)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_C_COMPILER= /usr/bin/gcc)
    include_directories(
        /usr/local/include
        /usr/include/eigen3
        /usr/include/x86_64-linux-gnu/c++/11
        /usr/include/c++/11
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
    link_directories(
        /usr/local/lib
        /usr/include/eigen3
        /usr/include/x86_64-linux-gnu/c++/11
        /usr/include/c++/11
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
endif(UNIX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

add_subdirectory(pybind11)
find_package(OpenMP REQUIRED)


# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /theoryfs2/ds/amwalla3/miniconda3/envs/qcn/include/python3.8 /usr/include/eigen3)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /usr/include/eigen3)
pybind11_add_module(dispersion src/binder.cpp src/disp.cpp src/r4r2.cpp)
# target_link_libraries(dispersion PRIVATE disp)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /theoryfs2/ds/amwalla3/miniconda3/envs/qcn/include/python3.8 /usr/include/eigen3)
# add_library(disp STATIC src/binder.cpp src/disp.cpp src/r4r2.cpp)

# target_include_directories(disp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /theoryfs2/ds/amwalla3/miniconda3/envs/qcn/include/python3.8 /usr/include/eigen3)
# set_target_properties(disp PROPERTIES POSITION_INDEPENDENT_CODE ON INTERFACE pybind11/include)
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()
