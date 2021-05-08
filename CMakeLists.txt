cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME ml_library)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

add_executable(library src/main.cpp src/Linear_model.cpp src/MLP_model.cpp src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h)
#add_library(${PROJECT_NAME} SHARED src/Linear_model.cpp src/MLP_model.cpp src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h)
