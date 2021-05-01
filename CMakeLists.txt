cmake_minimum_required(VERSION 3.19)
project(library)

set(CMAKE_CXX_STANDARD 14)

add_executable(library main.cpp Linear_model.cpp MLP_model.cpp)