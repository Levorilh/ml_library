#include "headers/main.h"
//#include "headers/tests/mlp_model_tests.h"
#include "headers/tests/linear_model_test.h"

int main() {
    srand(time(nullptr));

//    test_regression_linear();
//    test_classification_linear();

    test_regression_mlp();
//    test_classification_mlp();

//    test_multiclassification_mlp();

    return 0;
}