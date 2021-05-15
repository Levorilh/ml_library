#include "headers/main.h"

#include "headers/tests/mlp_model_tests.h"

int main(int argc, char** argv) {
    srand(time(nullptr));

    //test_regression_linear();
    //test_classification_linear();

    //test_regression_mlp();
    test_classification_mlp();

//    test_multiclassification_mlp();

    return 0;
}