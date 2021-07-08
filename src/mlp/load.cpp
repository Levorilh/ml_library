//
// Created by N on 19/06/2021.
//

#include "../headers/mlp/MLP.h"

DLLEXPORT MLP* load_mlp_model(char * path){
    int maxLength = 250;
    FILE *fp;
    fp = fopen(path , "r");
    if(!fp){
        return nullptr;
    }
    int npl_max = 0;
    char* model_to_string = (char*)malloc(sizeof(char) * maxLength);
    int * len = (int*)malloc(sizeof(int));

    //d_lenght
    fgets(model_to_string, maxLength,fp);
    int d_length = strtol(model_to_string, nullptr, 10);

    //d
    fgets(model_to_string, maxLength,fp);
    char ** s = split(model_to_string, len);
    int* d = (int*)malloc(sizeof(int) * (*len));
    for(int i =0; i< *len; i++){
        d[i] = atoi(s[i]);
        if(d[i] > npl_max){
            npl_max = d[i];
        }
        free(s[i]);
    }
    free(s);

    //X
    auto** X = (float**)malloc(sizeof(float*) * d_length);
    for(int i = 0; i<d_length; i++){
        X[i] = (float*)malloc(sizeof(float) * (*len + 1));
        for(int j = 0; j < (*len + 1); j++){
            X[i][j] = 0;
        }
    }

    //deltas
    float** deltas = (float**)malloc(sizeof(float*) * d_length);
    for(int i = 0; i<d_length; i++){
        deltas[i] = (float*)malloc(sizeof(float) * (*len));
        for(int j = 0; j < (*len); j++){
            deltas[i][j] = 0;
        }
    }

    //W
    float*** W = (float***)malloc(sizeof(float**) * d_length);
    for(int l = 0; l < d_length; l++){
        if (l == 0){
            W[l] = (float **)malloc(sizeof(float*)*npl_max);
            continue;
        } else {
            W[l] = (float **) malloc(sizeof(float *) * (d[l - 1] + 1));
        }
        for(int i = 0; i < d[l-1] + 1; i++){
            W[l][i] = (float*)malloc(sizeof(float) * (d[l] + 1));

            for(int j = 0; j < d[l] + 1; j++){
                model_to_string = fgets(model_to_string, maxLength,fp);
                W[l][i][j] = strtof(model_to_string, nullptr);
            }
        }
    }

    MLP * result = (MLP *)(malloc(sizeof(MLP)));
    result->d_length = d_length;
    result->d = d;
    result->deltas = deltas;
    result->X = X;
    result->W = W;

    free(len);
    free(model_to_string);
    fclose(fp);

    return result;
}
