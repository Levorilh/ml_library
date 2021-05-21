#include "../headers/mlp/destruct.h"


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
