#include "../headers/mlp/MLP.h"


void forward_pass(MLP* mlp,const float* sample_inputs,const bool is_classification)
{
    int L = mlp->d_length - 1;

    for(int j = 1; j < mlp->d[0] + 1; j++)
        mlp->X[0][j] = sample_inputs[j - 1];

    for(int l = 1; l < L + 1; l++){
        for(int j = 1; j < mlp->d[l] + 1; j++){
            float sum_result = 0.0;
            for(int i = 0; i < mlp->d[l - 1] + 1; i++){
                sum_result += mlp->W[l][i][j] * mlp->X[l - 1][i];
            }
            mlp->X[l][j] = sum_result;
            if (is_classification || l < L){
                mlp->X[l][j] = tanh(mlp->X[l][j]);
            }
        }
    }
}

