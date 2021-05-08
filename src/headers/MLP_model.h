#ifndef LIBRARY_MLP_MODEL_H
#define LIBRARY_MLP_MODEL_H


    #include <cstdio>
    #include <cstdlib>
    #include "../../Eigen/Dense"

    #include "main.h"

    using Eigen::MatrixXd;
    using namespace std;

    DLLEXPORT struct MLP;
    DLLEXPORT MLP * create_mlp_model(int* npl, int npl_length);

    DLLEXPORT void destroy_mlp_prediction(float * prediction);
    DLLEXPORT void destroy_mlp_model(MLP * mlp);
    DLLEXPORT void destroy_mlp_X_model(MLP *mlp);
    DLLEXPORT void destroy_mlp_deltas_model(MLP *mlp);
    DLLEXPORT void destroy_mlp_W_model(MLP *mlp);

    DLLEXPORT void forward_pass(MLP * mlp, float * sample_inputs, bool is_classification);
    DLLEXPORT void train_stochastic_gradient_backpropagation(MLP * mlp,
                                                         float * flattened_dataset_inputs,
                                                         int samples_count,
                                                         float * flattened_dataset_expected_outputs,
                                                         bool is_classification,
                                                         float alpha = 0.001,
                                                         int iterations_count = 100000);
    DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP * mlp,
                                                                                      float * flattened_dataset_inputs,
                                                                                      int samples_count,
                                                                                      float * flattened_dataset_expected_outputs,
                                                                                      float alpha= 0.01,
                                                                                      int iterations_count = 100000);

    DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP * mlp,
                                                                                  float * flattened_dataset_inputs,
                                                                                  int samples_count,
                                                                                  float * flattened_dataset_expected_outputs,
                                                                                  float alpha= 0.001,
                                                                                  int iterations_count = 10000);

    DLLEXPORT float * predict_mlp_model_regression(MLP * mlp, float * sample_input);
    DLLEXPORT float * predict_mlp_model_classification(MLP * mlp, float * sample_input);


#endif //LIBRARY_MLP_MODEL_H

