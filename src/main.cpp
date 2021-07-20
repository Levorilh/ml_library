#include "headers/tests/mlp_model_test.h"
#include "headers/tests/linear_model_test.h"
#include "headers/tests/rbfn_model_test.h"
#include "headers/tests/svm_model_test.h"



int main() {

    init_random();
//    test_regression_linear();
//    test_classification_linear();

//    test_regression_mlp();
//    test_classification_mlp();
//
//    test_multiclassification_mlp();

//    test_regression_linear_tricky();

//    test_save_mlp_model();
//    test_load_linear();

//    test_get_distance_rbfn();
//    test_kmeans_rbfn();
//    test_rbfn();
    test_save_load_rbf();

//    test_svm_model();
//    test_img_mlp_model();

    return 0;
}


