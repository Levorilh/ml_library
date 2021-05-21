#ifndef ML_LIBRARY_MLP_TRAIN_H
#define ML_LIBRARY_MLP_TRAIN_H

#include "MLP.h"
#include "create.h"

void train_stochastic_gradient_backpropagation(MLP * mlp,
                                               const float flattened_dataset_inputs[],
                                               const int samples_count,
                                               const float flattened_dataset_expected_outputs[],
                                               const bool is_classification,
                                               const float alpha = 0.001,
                                               const int iterations_count = 100000);

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

#endif //ML_LIBRARY_TRAIN_H
