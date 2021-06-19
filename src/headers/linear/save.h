//
// Created by N on 09/06/2021.
//

#ifndef ML_LIBRARY_SAVE_H
#define ML_LIBRARY_SAVE_H

#include "../main.h"

#include <iostream>
#include <fstream>

DLLEXPORT void save_linear_model(float* model, int input_dim, char * path);

#endif //ML_LIBRARY_SAVE_H
