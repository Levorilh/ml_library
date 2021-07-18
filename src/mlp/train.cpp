#include "../headers/mlp/MLP.h"

void train_stochastic_gradient_backpropagation(MLP * mlp,
                                               const float* flattened_dataset_inputs,
                                               const int samples_count,
                                               const float* flattened_dataset_expected_outputs,
                                               const bool is_classification,
                                               const float alpha,
                                               const int iterations_count)
{
    const int input_dim = mlp->d[0];
    const int output_dim = mlp->d[mlp->d_length - 1];
    const int L = mlp->d_length - 1;

    const int ONE_PERCENT = iterations_count/100;
    const int FIVE_PERCENT = ONE_PERCENT * 5;

    for(int it = 0; it < iterations_count; it++){
        if(it % FIVE_PERCENT == 1){
            cout << "train " << it/FIVE_PERCENT * 5 << "% complete " << endl;
        }
        int k = rand() % samples_count;

        auto * sample_input = (float *)malloc(sizeof(float) * input_dim);
        for (int index = 0; index < input_dim; index ++) {
            sample_input[index] = flattened_dataset_inputs[k * input_dim + index];
        }

        auto * sample_expected_output = (float *)malloc(sizeof(float) * output_dim);
        for (int index = 0; index < output_dim; index ++)
            sample_expected_output[index] = flattened_dataset_expected_outputs[k * output_dim + index];

        forward_pass(mlp, sample_input, is_classification);

        for(int j = 1; j < mlp->d[L] + 1; j++){
            mlp->deltas[L][j] = mlp->X[L][j] - sample_expected_output[j - 1];
            if (is_classification)
                mlp->deltas[L][j] *= (1 - mlp->X[L][j] * mlp->X[L][j]);
        }

        for(int l = L; l > 0; l--){
            for(int i = 1; i < mlp->d[l - 1] + 1; i++){
                float sum_result = 0.0;
                for(int j = 1; j < mlp->d[l] + 1; j++){
                    sum_result += mlp->W[l][i][j] * mlp->deltas[l][j];
                }
                mlp->deltas[l - 1][i] = (1 - mlp->X[l-1][i] * mlp->X[l-1][i]) * sum_result;

            }
        }

        for(int l = 1; l < L + 1; l++)
            for(int i = 0; i < mlp->d[l-1] + 1; i++)
                for(int j = 1; j < mlp->d[l] + 1; j++) {
                    mlp->W[l][i][j] -= alpha * mlp->X[l - 1][i] * mlp->deltas[l][j];
                    //cout << "\tW[" << l << "][" << i << "][" << j << "] : " << mlp->W[l][i][j] << endl;
                }

        free(sample_input);
        free(sample_expected_output);
    }


    /*
    cout << "Weights changed: " << endl;
    for(int l = 1; l < L + 1; l++)
        for(int i = 0; i < mlp->d[l-1] + 1; i++)
            for(int j = 1; j < mlp->d[l] + 1; j++) {
                cout << "W[" << l << "][" << i << "][" << j << "] : " << mlp->W[l][i][j] << endl;
            }
    cout << endl;
    */

}



void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP * mlp,
                                                                        const float* flattened_dataset_inputs,
                                                                        const int samples_count,
                                                                        const float* flattened_dataset_expected_outputs,
                                                                        const float alpha,
                                                                        const int iterations_count) {
    train_stochastic_gradient_backpropagation(mlp,
                                              flattened_dataset_inputs,
                                              samples_count,
                                              flattened_dataset_expected_outputs,
                                              true,
                                              alpha,
                                              iterations_count);
}


void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP * mlp,
                                                                    const float* flattened_dataset_inputs,
                                                                    const int samples_count,
                                                                    const float* flattened_dataset_expected_outputs,
                                                                    const float alpha,
                                                                    const int iterations_count) {
    train_stochastic_gradient_backpropagation(mlp,
                                              flattened_dataset_inputs,
                                              samples_count,
                                              flattened_dataset_expected_outputs,
                                              false,
                                              alpha,
                                              iterations_count);
}