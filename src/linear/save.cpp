#include "../headers/linear/save.h"


DLLEXPORT void save_linear_model(float* model, int input_dim, char *path){
    string model_to_string = "";
    for (int i =0; i<input_dim + 1; i++){// +1 pour le biai
        if(i>0){
            model_to_string = model_to_string + ";";
        }
        model_to_string = model_to_string + std::to_string(model[i]);
    }

    FILE *save_file;
    fopen_s(&save_file, path , "r");
    fwrite(  model_to_string.c_str() , sizeof(char) , model_to_string.size() ,save_file);
    fclose(save_file);
}

