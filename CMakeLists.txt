cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME ml_library)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(MAIN src/main.cpp src/headers/main.h)

set(UTILS src/utils.cpp src/headers/utils.h)

set(mlp_SOURCE_FILES src/mlp/MLP.cpp src/mlp/predict.cpp       src/mlp/train.cpp         src/mlp/destruct.cpp       src/mlp/create.cpp  src/mlp/save.cpp src/mlp/load.cpp)
set(mlp_HEADER_FILES src/headers/mlp/predict.h src/headers/mlp/train.h   src/headers/mlp/destruct.h src/headers/mlp/create.h    src/headers/mlp/save.h src/headers/mlp/load.h)

set(MLP ${mlp_SOURCE_FILES} ${mlp_HEADER_FILES})

set(linear_SOURCE_FILES     src/linear/create.cpp           src/linear/predict.cpp          src/linear/train.cpp        src/linear/destroy.cpp          src/linear/load.cpp         src/linear/save.cpp    )
set(linear_HEADER_FILES     src/headers/linear/create.h     src/headers/linear/predict.h    src/headers/linear/train.h  src/headers/linear/destruct.h   src/headers/linear/load.h   src/headers/linear/save.h   )

set(LINEAR ${linear_SOURCE_FILES} ${linear_HEADER_FILES}  )

set(svm_SOURCE_FILES    src/svm/create.cpp          )
set(svm_HEADER_FILES    src/headers/svm/create.h    )

set(SVM ${svm_SOURCE_FILES} ${svm_HEADER_FILES}  )

set(rbfn_SOURCE_FILES   src/rbfn/rbf.cpp        src/rbfn/Centroid.cpp)
set(rbfn_HEADER_FILES   src/headers/rbfn/rbf.h  src/headers/rbfn/Centroid.h)
set(RBFN ${rbfn_SOURCE_FILES} ${rbfn_HEADER_FILES})



set(TEST_FILES_LINEAR src/tests/linear_model_tests.cpp src/headers/tests/linear_model_test.h)
set(TEST_FILES_MLP src/tests/mlp_model_tests.cpp src/headers/tests/mlp_model_test.h)
set(TEST_FILES_RBFN src/tests/rbfn_model_tests.cpp src/headers/tests/rbfn_model_test.h)
set(TEST_FILES_SVM src/tests/svm_model_test.cpp src/headers/tests/svm_model_test.h)
set(TEST_FILES ${TEST_FILES_LINEAR} ${TEST_FILES_MLP} ${TEST_FILES_RBFN} ${TEST_FILES_SVM})

add_executable(${PROJECT_NAME} ${MAIN} ${LINEAR} ${MLP}  ${RBFN} ${UTILS} ${TEST_FILES} ${SVM})
#add_library(${PROJECT_NAME} SHARED ${LINEAR} ${MLP} ${UTILS} ${RBFN})
