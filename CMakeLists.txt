cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME ml_library)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/main.cpp src/Linear_model.cpp src/MLP_model.cpp)
set(HEADER_FILES src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h)

set(TEST_FILES src/tests/mlp_model_tests.cpp src/headers/tests/mlp_model_tests.h)
#add_executable(library  ${SOURCE_FILES} ${HEADER_FILES} ${TEST_FILES} )
add_library(${PROJECT_NAME} SHARED src/Linear_model.cpp src/MLP_model.cpp src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h src/utils.cpp)
