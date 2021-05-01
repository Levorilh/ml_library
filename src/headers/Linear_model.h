#ifndef LIBRARY_LINEAR_MODEL_H
#define LIBRARY_LINEAR_MODEL_H


    #include <iostream>
    #include <cstdio>
    #include <cstdlib>
    #include <ctime>
    #include "../../Eigen/Dense"

    #include "main.h"

    using Eigen::MatrixXd;
    using namespace std;

    DLLEXPORT float *create_linear_model(int);

    DLLEXPORT void destroy_linear_model(float *);

    DLLEXPORT void train_regression_pseudo_inverse_linear_model(float *model,
                                                      int input_dim,
                                                      float *flattened_dataset_inputs,
                                                      int samples_count,
                                                      float *flattened_dataset_expected_outputs);

    DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float *model,
                                                           int input_dim,
                                                           float *flattened_dataset_inputs,
                                                           int samples_count,
                                                           float *flattened_dataset_expected_outputs,
                                                           float alpha = 0.001,
                                                           int iterations_count = 1000);

#endif //LIBRARY_LINEAR_MODEL_H
