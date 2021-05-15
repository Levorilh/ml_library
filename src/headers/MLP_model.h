#ifndef LIBRARY_MLP_MODEL_H
#define LIBRARY_MLP_MODEL_H


    #include <cstdio>
    #include <cstdlib>
    #include "../../Eigen/Dense"

    #include "main.h"

    using Eigen::MatrixXd;
    using namespace std;

    typedef struct MLP_t{
        int d_length;
        int *d;
        float** X;
        float** deltas;
        float*** W;
    } MLP;

    DLLEXPORT MLP;
    DLLEXPORT MLP* create_mlp_model(const int npl[], const int npl_length);

    DLLEXPORT void destroy_mlp_prediction(float * prediction);
    DLLEXPORT void destroy_mlp_model(MLP * mlp);
    void destroy_mlp_X_model(MLP *mlp);
    void destroy_mlp_deltas_model(MLP *mlp);
    void destroy_mlp_W_model(MLP *mlp);

    void forward_pass(MLP* mlp, const float sample_inputs[], const bool is_classification[]);
    void train_stochastic_gradient_backpropagation(MLP* mlp,
                                                         float* flattened_dataset_inputs,
                                                         int samples_count,
                                                         float* flattened_dataset_expected_outputs,
                                                         bool is_classification,
                                                         float alpha = 0.001,
                                                         int iterations_count = 100000);
    DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP* mlp,
                                                                                      const float flattened_dataset_inputs[],
                                                                                      const int samples_count,
                                                                                      const float flattened_dataset_expected_outputs[],
                                                                                      const float alpha= 0.01,
                                                                                      const int iterations_count = 100000);

    DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP* mlp,
                                                                                  const float flattened_dataset_inputs[],
                                                                                  const int samples_count,
                                                                                  const float flattened_dataset_expected_outputs[],
                                                                                  const float alpha= 0.001,
                                                                                  const int iterations_count = 10000);

    DLLEXPORT float * predict_mlp_model_regression(MLP* mlp, float* sample_input);
    DLLEXPORT float * predict_mlp_model_classification(MLP* mlp, float* sample_input);


#endif //LIBRARY_MLP_MODEL_H

