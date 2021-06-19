//
// Created by N on 19/06/2021.
//

#include <xutility>
#include "../headers/mlp/MLP.h"

DLLEXPORT MLP* load_mlp_model(char * path){
    int maxLength = 250;
    FILE *fp;
    fp = fopen(path , "r");
    char* model_to_string = (char*)malloc(sizeof(maxLength));
    int * len = (int*)malloc(sizeof(int));

    //d_lenght
    fgets(model_to_string, maxLength,fp);
    int d_length = atoi(model_to_string);

    //d
    fgets(model_to_string, maxLength,fp);
    char ** s = split(model_to_string, len);
    int* d = (int*)malloc(sizeof(int) * (*len));
    for(int i =0; i< *len; i++){
        d[i] = atoi(s[i]);
        free(s[i]);
    }
    free(s);

    //X
    /*float** X = (float**)malloc(sizeof(float*) * d_length);
    for(int i = 0; i<d_length; i++){
        fgets(model_to_string, maxLength,fp);
        char ** s = split(model_to_string, len);
        float* X2 = (float*)malloc(sizeof(float) * (*len));
        X[i] = X2;
        for(int j = 0; j < (*len); j++){
            X[i][j] = atof(s[j]);
            free(s[j]);
        }
        free(s);
    }*/

    //deltas
    /*float** deltas = (float**)malloc(sizeof(float*) * d_length);
    for(int i = 0; i<d_length; i++){
        fgets(model_to_string, maxLength,fp);
        char ** s = split(model_to_string, len);
        float* deltas2 = (float*)malloc(sizeof(float) * (*len));
        X[i] = deltas2;
        for(int j = 0; j < (*len); j++){
            X[i][j] = atof(s[j]);
            free(s[j]);
        }
        free(s);
    }*/

    //W
    float*** W = (float***)malloc(sizeof(float**) * d_length);
    fgets(model_to_string, maxLength,fp);
    char ** flatenned_W = split(model_to_string, len);
    int cpt = 0;

    for(int l = 1; l < d_length; l++){
        float** W2 = (float**)malloc(sizeof(float*) * d[l-1] + 1);
        W[l] = W2;

        for(int i = 0; i < d[l-1] + 1; i++){
            float* W3 = (float*)malloc(sizeof(float) * d[l] + 1);
            W[l][i] = W3;

            for(int j = 0; j < d[l] + 1; j++){
                W[l][i][j] = atof(flatenned_W[cpt]);
                cpt++;
            }
        }
    }

    MLP * result = (MLP *)(malloc(sizeof(MLP)));
    result->d_length = d_length;
    result->d = d;
    //result->deltas = deltas;
    //result->X = X;
    result->W = W;

    free(len);
    //free(model_to_string);
    fclose(fp);

    return NULL;
}
