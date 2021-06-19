#include "../headers/mlp/MLP.h"

DLLEXPORT void save_mlp_model(MLP* model, char* path){
    FILE* fp = fopen( path, "w" );
    //fprintf(fp, "-d_lenght-\n");
    fprintf(fp, "%d\n",model->d_length);

    //fprintf(fp, "-d-\n");
    for(int i = 0; i < model->d_length; i++){
        if(i>0)
            fprintf(fp, ";");
        fprintf(fp, "%d",model->d[i]);
    }
    fprintf(fp, "\n");

    //fprintf(fp, "-X-\n");
    /*for(int i = 0; i < model->d_length; i++){
        for(int j = 0; j < model->d[i] + 1; j++){ //+1 pour le biai
            if(j>0)
                fprintf(fp, ";");
            fprintf(fp, "%d",model->X[i][j]);
        }
        fprintf(fp, "\n"); // on represente ne 2d
    }

    fprintf(fp, "-deltas-\n");
    for(int i = 0; i < model->d_length; i++){
        for(int j = 0; j < model->d[i] + 1; j++){ //+1 pour le biai
            if(j>0)
                fprintf(fp, ";");
            fprintf(fp, "%d",model->deltas[i][j]);
        }
        fprintf(fp, "\n"); // on represente ne 2d
    }*/

    //fprintf(fp, "-W-\n"); // 3d difficile a représenter donc en flattened
    for(int l = 1; l < model->d_length; l++){ // commence par 1 ignore les biais
        for(int i = 0; i < model->d[l-1] + 1; i++){
            for(int j =0; j < model->d[l] + 1; j++){
                if(l+i+j>1)
                    fprintf(fp, ";");
                fprintf(fp, "%d",model->W[l][i][j]);
            }
        }
    }
    fclose(fp);
}