//
// Created by N on 01/05/2021.
//

#include <cstdio>
#include <cstdlib>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using namespace std;

float *create_mlp_model(int);
float *forward_pass(float sample_inputs, bool is_classification);
float *train_stochastic_gradient_descent(float flattened_dataset_inputs,
                                        float flattened_dataset_expected_outputs,
                                        bool is_classification,
                                        float alpha = 0.001,
                                        int iterations_count = 10000);

int main() {

}