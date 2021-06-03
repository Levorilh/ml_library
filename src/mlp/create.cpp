#include "../headers/mlp/MLP.h"

MLP * create_mlp_model(int* npl,const int npl_length){
    int npl_max = npl[0];
    for(int i = 1; i < npl_length; i++){  // findMax(array)
        if (npl_max < npl[i]){
            npl_max = npl[i];
        }
    }

    int* d = (int*)malloc(sizeof(int) * npl_length);
    for(int i = 0; i< npl_length; i++){
        d[i] = npl[i];
    }

    auto *** W = (float ***)(malloc(sizeof(float**) * npl_length));
    auto ** X = (float **)(malloc(sizeof(float*) * npl_length));
    auto ** deltas = (float **)(malloc(sizeof(float *) * npl_length));


    for(int l = 0; l < npl_length; l++){
        if (l == 0){
            W[l] = (float **)malloc(sizeof(float*)*npl_max);
            continue;
        } else
            W[l] = (float **)malloc(sizeof(float*)*(npl[l - 1] + 1));
        for(int i = 0; i < npl[l - 1] + 1; i++) {
            W[l][i] = (float *)malloc(sizeof(float)*(npl[l] + 1));
            for (int j = 0; j < npl[l] + 1; j++) {
                float x = ( ((float)(rand() % 2001)) / 1000.0) - 1.;
                //std::cout << "Weights initialisation: " << x << std::endl;
                W[l][i][j] = x;
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
    result->d = d;
    result->deltas = deltas;
    result->X = X;
    result->W = W;

    return result;
}