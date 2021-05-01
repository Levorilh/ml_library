#ifndef LIBRARY_MLP_MODEL_H
#define LIBRARY_MLP_MODEL_H


    #include <cstdio>
    #include <cstdlib>
    #include "../../Eigen/Dense"

    #include "main.h"

    using Eigen::MatrixXd;
    using namespace std;

    float *create_mlp_model(int);
    float *forward_pass(float sample_inputs, bool is_classification);
    float *train_stochastic_gradient_descent(float flattened_dataset_inputs,
                                             float flattened_dataset_expected_outputs,
                                             bool is_classification,
                                             float alpha = 0.001,
                                             int iterations_count = 10000);



#endif //LIBRARY_MLP_MODEL_H

