cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME ml_library)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(MAIN src/main.cpp src/headers/main.h)

set(mlp_SOURCE_FILES src/main.cpp src/mlp/Linear_model.cpp src/mlp/predict.cpp src/mlp/train.cpp src/mlp/destroy.cpp src/mlp/create.cpp)
set(mlp_HEADER_FILES src/headers/mlp/predict.h src/headers/mlp/train.h src/headers/mlp/destroy.h src/headers/mlp/create.h src/headers/mlp/Linear_model.h)

set(MLP ${mlp_SOURCE_FILES} ${mlp_HEADER_FILES})

set(linear_SOURCE_FILES     src/linear/create.cpp           src/linear/predict.cpp          src/linear/train.cpp        src/linear/destroy.cpp)
set(linear_HEADER_FILES     src/headers/linear/create.h     src/headers/linear/predict.h    src/headers/linear/train.h  src/headers/linear/destroy.h)

set(LINEAR ${linear_SOURCE_FILES} ${linear_HEADER_FILES})

set(TEST_FILES src/tests/linear_model_tests.cpp src/headers/tests/linear_model_test.h)


set(COMPILATION_MODE 1)


if( COMPILATION_MODE EQUAL 1 )

    add_executable(library ${MAIN} ${LINEAR} ${TEST_FILES})
    #add_library(${PROJECT_NAME} SHARED src/Linear_model.cpp src/MLP_model.cpp src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h src/utils.cpp)
elseif( COMPILATION_MODE EQUAL 2)

    add_executable(${PROJECT_NAME} ${MAIN} ${MLP})
    #add_library(${PROJECT_NAME} SHARED src/Linear_model.cpp src/MLP_model.cpp src/headers/MLP_model.h src/headers/main.h src/headers/Linear_model.h src/utils.cpp)
else()
    message(WARNING "$COMPILATION_MODE not set")
endif()