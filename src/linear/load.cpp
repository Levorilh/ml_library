#include "../headers/linear/load.h"

float* load_linear_model(char path[]){
    ifstream MyReadFile(path);
    string model_to_string;
    getline (MyReadFile, model_to_string);
    MyReadFile.close();

    //TODO : error
    return NULL;
}