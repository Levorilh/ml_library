#include "../headers/mlp/MLP.h"

float* predict_mlp_model_regression(MLP * mlp, float * sample_input){
    forward_pass(mlp,sample_input,false);

    auto * result = (float *)malloc(sizeof(float) * mlp->d_length - 1);
    for(int i = 1; i < (mlp->d[mlp->d_length - 1] + 1); i++){
        result[i - 1] = mlp->X[mlp->d_length - 1][i];
    }

    return result;
}

float* predict_mlp_model_classification(MLP* mlp, float * sample_input){
    forward_pass(mlp,sample_input,true);

    auto * result = (float *)malloc(sizeof(float) * (mlp->d_length - 1));

    for(int i = 1; i < (mlp->d[mlp->d_length - 1] + 1); i++){
        result[i - 1] = mlp->X[mlp->d_length - 1][i];
    }
    return result;
}
