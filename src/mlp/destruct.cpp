#include "../headers/mlp/MLP.h"
void destroy_mlp_prediction(const float * prediction){
    printf("pred : %p \n", prediction);
    delete prediction;
}

void destroy_mlp_model(MLP * mlp){
    printf("X : %p\n" , mlp->X);
    destroy_mlp_W_model(mlp);
    destroy_mlp_X_model(mlp);
    destroy_mlp_deltas_model(mlp);
    free(mlp->d);
    printf("model : %p\n" , mlp);
    free(mlp);
    printf("end frees\n");
}

void destroy_mlp_X_model(MLP *mlp){
    for(int l = 0; l < mlp->d_length; l++) {
        printf("X[%d] : %p\n" , l , mlp->X[l]);
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
