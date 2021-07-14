#ifndef ML_LIBRARY_MLP_MODEL_TESTS_H
#define ML_LIBRARY_MLP_MODEL_TESTS_H

#include "../mlp/MLP.h"
#include "../main.h"
#include "../linear/create.h"
#include "../mlp/save.h"

void test_regression_mlp();
void test_classification_mlp();
void test_multiclassification_mlp();
void test_save_mlp_model();
void test_img_mlp_model();

#endif //ML_LIBRARY_MLP_MODEL_TESTS_H
