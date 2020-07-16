cmake_minimum_required(VERSION 3.15)
project(DP_Shape_Match_func_nDim)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(DP_Shape_Match_SRVF_nDim
        src/DP_Shape_Match_SRVF_nDim.cpp
        src/functions.cpp
        src/functions.h
        src/shape.cpp
        src/shape.h)

add_executable(DP_Shape_Match_Coord_nDim
        src/DP_Shape_Match_Coord_nDim.cpp
        src/functions.cpp
        src/functions.h
        src/shape.cpp
        src/shape.h)
