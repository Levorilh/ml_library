#include "../headers/linear/load.h"

DLLEXPORT float* load_linear_model(char path[], int* input_dim){
    int maxLength = 100;
    FILE *fp;
    fp = fopen(path , "r");
    char* model_to_string = (char*)malloc(sizeof(maxLength));
    fgets(model_to_string, maxLength,fp);

    int * len = (int*)malloc(sizeof(int));
    char ** s = split(model_to_string, len);
    float* rslt = (float*)malloc(sizeof(float) * (*len));
    for(int i =0; i< *len; i++){
        rslt[i] = atoi(s[i]);
        free(s[i]);
    }

    *input_dim = (*len) - 1;

    free(len);
    free(s);
    //free(model_to_string);
    fclose(fp);

    return rslt;
}