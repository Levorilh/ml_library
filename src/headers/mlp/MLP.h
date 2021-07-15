#ifndef ML_LIBRARY_MLP_H
#define ML_LIBRARY_MLP_H

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../../Eigen/Dense"

using Eigen::MatrixXd;
using namespace std;

#include "../main.h"

class MLP_t{
public:
    int *d;
    float** X;
    float** deltas;
    float*** W;
    int d_length;
};
typedef class MLP_t MLP;


DLLEXPORT MLP* create_mlp_model(int* npl,const int npl_length);


DLLEXPORT float * predict_mlp_model_regression(MLP* mlp, float* sample_input);
DLLEXPORT float * predict_mlp_model_classification(MLP* mlp, float* sample_input);


void forward_pass(MLP* mlp,const float* sample_inputs,const bool is_classification);

void train_stochastic_gradient_backpropagation(MLP * mlp,
                                               const float* flattened_dataset_inputs,
                                               const int samples_count,
                                               const float* flattened_dataset_expected_outputs,
                                               const bool is_classification,
                                               const float alpha = 0.001,
                                               const int iterations_count = 100000);

DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP* mlp,
                                                                                  const float* flattened_dataset_inputs,
                                                                                  const int samples_count,
                                                                                  const float* flattened_dataset_expected_outputs,
                                                                                  const float alpha= 0.01,
                                                                                  const int iterations_count = 100000);

DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP* mlp,
                                                                              const float* flattened_dataset_inputs,
                                                                              const int samples_count,
                                                                              const float* flattened_dataset_expected_outputs,
                                                                              const float alpha= 0.001,
                                                                              const int iterations_count = 10000);



DLLEXPORT void destroy_mlp_prediction(const float * prediction);
DLLEXPORT void destroy_mlp_model(MLP * mlp);
void destroy_mlp_W_model(MLP *mlp);
void destroy_mlp_deltas_model(MLP *mlp);
void destroy_mlp_X_model(MLP *mlp);
DLLEXPORT void save_mlp_model(MLP* model,const char* path);
DLLEXPORT MLP* load_mlp_model(const char * path);


#endif //ML_LIBRARY_MLP_H
