#ifndef ML_LIBRARY_DESTRUCT_H
#define ML_LIBRARY_DESTRUCT_H

#include "create.h"
DLLEXPORT void destroy_mlp_prediction(float * prediction);
DLLEXPORT void destroy_mlp_model(MLP * mlp);
void destroy_mlp_W_model(MLP *mlp);


#endif //ML_LIBRARY_DESTRUCT_H
