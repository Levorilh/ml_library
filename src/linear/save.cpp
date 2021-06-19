#include "../headers/linear/save.h"


void save_linear_model(float* model, int input_dim, char path[]){
    string model_to_string = "";
    for (int i =0; i<input_dim + 1; i++){// +1 pour le biai
        if(i>0){
            model_to_string = model_to_string + ";";
        }
        model_to_string = model_to_string + std::to_string(model[i]);
    }

    ofstream save_file(path);
    save_file << model_to_string;
    save_file.close();
}

