#include "headers/MLP_model.h"
#include <math.h>

using Eigen::MatrixXd;
using namespace std;

void destroy_mlp_prediction(float * prediction){
    free(prediction);
}

void destroy_mlp_model(MLP * mlp){
    destroy_mlp_W_model(mlp);
    destroy_mlp_X_model(mlp);
    destroy_mlp_deltas_model(mlp);

    free(mlp->d);
    free(mlp);
}

void destroy_mlp_X_model(MLP *mlp){
    for(int l = 0; l < mlp->d_length; l++) {
        free(mlp->X[l]);
    }
    free(mlp->X);
}
void destroy_mlp_deltas_model(MLP *mlp){
    for(int l = 0; l < mlp->d_length; l++) {
        free(mlp->deltas[l]);
    }
    free(mlp->deltas);
}
void destroy_mlp_W_model(MLP *mlp){
    for(int l = 0; l < mlp->d_length; l++){
        if (l == 0) continue;
        for(int i = 0; i < mlp->d[l - 1] + 1; i++) {
            free(mlp->W[l][i]);
        }
        free(mlp->W[l]);
    }
    free(mlp->W);
}


MLP * create_mlp_model(int* npl,const int npl_length){
    int npl_max = npl[0];
    for(int i = 1; i < npl_length; i++){  // findMax(array)
        if (npl_max < npl[i]){
            npl_max = npl[i];
        }
    }

    // que faire des espaces non remplie ?...
    float *** W = (float ***)(malloc(sizeof(float**) * npl_length));     //[npl_length][npl_max][npl_max];
    float ** X = (float **)(malloc(sizeof(float*) * npl_length));        //[npl_length][npl_max];
    float ** deltas = (float **)(malloc(sizeof(float *) * npl_length));  //[npl_length][npl_max];


    for(int l = 0; l < npl_length; l++){
        if (l == 0){
            W[l] = (float **)malloc(sizeof(float*)*npl_max);
            continue;
        } else
            W[l] = (float **)malloc(sizeof(float*)*(npl[l - 1] + 1));
        for(int i = 0; i < npl[l - 1] + 1; i++) {
            W[l][i] = (float *)malloc(sizeof(float)*(npl[l] + 1));
            for (int j = 0; j < npl[l] + 1; j++) {
                float x = ((rand() % 2001) / 1000.0) - 1.;
                W[l][i][j] = x;
                //cout << "l : " << l << "/ i : " << i << " / j : " << j << "/ W : " << W[l][i][j] << endl;
            }
        }
    }

    for(int l = 0; l < npl_length; l++) {
        X[l] = (float *)malloc(sizeof(float) * (npl[l] + 1));
        for (int j = 0; j < npl[l] + 1; j++)
            X[l][j] = j == 0 ? 1. : 0.;
    }

    for(int l = 0; l < npl_length; l++) {
        deltas[l] = (float *)malloc(sizeof(float) * (npl[l] + 1));
        for (int j = 0; j < npl[l] + 1; j++)
            deltas[l][j] = 0.;
    }

    MLP * result = (MLP *)(malloc(sizeof(MLP)));
    result->d_length = npl_length;
    result->d = npl;
    result->deltas = deltas;
    result->X = X;
    result->W = W;

    return result;
}


void forward_pass(MLP * mlp,const float sample_inputs[],const bool is_classification){
    int L = mlp->d_length - 1;

    for(int j = 1; j < mlp->d[0] + 1; j++)
        mlp->X[0][j] = sample_inputs[j - 1];

    for(int l = 1; l < L + 1; l++)
        for(int j = 1; j < mlp->d[l] + 1; j++){
            float sum_result = 0.0;
            for(int i = 0; i < mlp->d[l - 1] + 1; i++)
                sum_result += mlp->W[l][i][j] * mlp->X[l - 1][i];
            mlp->X[l][j] = sum_result;
            if (is_classification || l < L)
                mlp->X[l][j] = tanh(mlp->X[l][j]);
        }
}

void train_stochastic_gradient_backpropagation(MLP * mlp,
                                                  const float flattened_dataset_inputs[],
                                                  const int samples_count,
                                                  const float flattened_dataset_expected_outputs[],
                                                  const bool is_classification,
                                                  const float alpha,
                                                  const int iterations_count){
    const int input_dim = mlp->d[0];
    const int output_dim = mlp->d[mlp->d_length - 1];
    const int L = mlp->d_length - 1;

    /*
    cout << endl;

    cout << "Weights initialisation: " << endl;
    for(int l = 1; l < L + 1; l++)
        for(int i = 0; i < mlp->d[l-1] + 1; i++)
            for(int j = 1; j < mlp->d[l] + 1; j++) {
                cout << "W[" << l << "][" << i << "][" << j << "] : " << mlp->W[l][i][j] << endl;
            }
    cout << endl;
    */
    for(int it = 0; it < iterations_count; it++){
        int k = rand() % samples_count;

        float * sample_input = (float *)malloc(sizeof(float) * input_dim);
        for (int index = 0; index < input_dim; index ++) {
            sample_input[index] = flattened_dataset_inputs[k * input_dim + index];
        }

        float * sample_expected_output = (float *)malloc(sizeof(float) * output_dim);
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
                    // cout << "W[" << l << "][" << i << "][" << j << "] : " << mlp->W[l][i][j] << endl;
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

float* predict_mlp_model_regression(MLP * mlp, float * sample_input){
    forward_pass(mlp,sample_input,false);

    float * result = (float *)malloc(sizeof(float) * mlp->d_length - 1);
    for(int i = 1; i < (mlp->d[mlp->d_length - 1] + 1); i++){
        result[i - 1] = mlp->X[mlp->d_length - 1][i];
    }
    return result;
}

float* predict_mlp_model_classification(MLP * mlp, float * sample_input){
    forward_pass(mlp,sample_input,true);

    float * result = (float *)malloc(sizeof(float) * (mlp->d_length - 1));
    for(int i = 1; i < (mlp->d[mlp->d_length - 1] + 1); i++){
        result[i - 1] = mlp->X[mlp->d_length - 1][i];
    }
    return result;
}


void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP * mlp,
                                                                       const float flattened_dataset_inputs[],
                                                                       const int samples_count,
                                                                       const float flattened_dataset_expected_outputs[],
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
                                                                        float * flattened_dataset_inputs,
                                                                        int samples_count,
                                                                        float * flattened_dataset_expected_outputs,
                                                                        float alpha,
                                                                        int iterations_count) {
    train_stochastic_gradient_backpropagation(mlp,
                                              flattened_dataset_inputs,
                                              samples_count,
                                              flattened_dataset_expected_outputs,
                                              false,
                                              alpha,
                                              iterations_count);
}