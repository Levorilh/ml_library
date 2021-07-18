#include "../headers/mlp/MLP.h"
void destroy_mlp_prediction(const float * prediction){
    delete prediction;
}

void destroy_mlp_model(MLP * mlp){
    if(mlp == nullptr){
        return;
    }
    destroy_mlp_W_model(mlp);
    destroy_mlp_X_model(mlp);
    destroy_mlp_deltas_model(mlp);
    free(mlp->d);
    free(mlp);
    printf("end frees\n");
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
        if (l == 0){
            free(mlp->W[l]);
            continue;
        }
        for(int i = 0; i < mlp->d[l - 1] + 1; i++) {
            free(mlp->W[l][i]);
        }
        free(mlp->W[l]);
    }
    free(mlp->W);
}
