cmake_minimum_required(VERSION 3.10)

project(DISPERSION VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(pybind11)
# set(pybind11_DIR pybind11)
# find_package(pybind11 REQUIRED)

if(APPLE)
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
endif()
if(LINUX)
    # set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
    # set(CMAKE_C_COMPILER= "/usr/bin/g++-11")
    include_directories(/usr/local/include
        /usr/include/eigen3
        /usr/include
        /usr/lib
        pybind11/include
        src
        )
    link_directories(/usr/local/lib
        /usr/include/eigen3
        /usr/include
        /usr/lib
        pybind11/include
        src
        )
endif()

find_package(OpenMP REQUIRED)


pybind11_add_module(dispersion src/binder.cpp src/disp.cpp)
# target_link_libraries(dispersion PRIVATE disp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /theoryfs2/ds/amwalla3/miniconda3/envs/qcn/include/python3.8)
add_library(disp STATIC src/binder.cpp src/disp.cpp)
target_include_directories(disp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include /theoryfs2/ds/amwalla3/miniconda3/envs/qcn/include/python3.8)
# set_target_properties(disp PROPERTIES POSITION_INDEPENDENT_CODE ON INTERFACE pybind11/include)
