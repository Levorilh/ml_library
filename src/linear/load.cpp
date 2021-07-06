#include "../headers/linear/load.h"

DLLEXPORT float* load_linear_model(char* path, int* input_dim){
    int maxLength = 10000;
    //todo replace maxLength with actual length of file (SEEK_END blablabla)
    FILE *fp = fopen(path , "r");

    char* model_to_string = (char*)malloc(sizeof(char)* maxLength);
    fgets(model_to_string, maxLength,fp);

    char ** s = split(model_to_string, input_dim);
    auto rslt = (float*)malloc(sizeof(float) * (*input_dim));
    for(int i =0; i< *input_dim; i++){
        rslt[i] = strtof(s[i],nullptr);
        free(s[i]);
    }

    free(s);
    free(model_to_string);
    fclose(fp);

    return rslt;
}