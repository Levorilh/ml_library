#ifndef ML_LIBRARY_LINEAR_TRAIN_H
#define ML_LIBRARY_LINEAR_TRAIN_H

#include "../main.h"
#include "create.h"
#include "predict.h"

DLLEXPORT void train_regression_pseudo_inverse_linear_model(float *model,
                                                            const int input_dim,
                                                            const float flattened_dataset_inputs[],
                                                            const int samples_count,
                                                            const float flattened_dataset_expected_outputs[]);

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float *model,
                                                                 int input_dim,
                                                                 float *flattened_dataset_inputs,
                                                                 int samples_count,
                                                                 float *flattened_dataset_expected_outputs,
                                                                 float alpha = 0.001,
                                                                 int iterations_count = 1000);

#endif //ML_LIBRARY_TRAIN_H
