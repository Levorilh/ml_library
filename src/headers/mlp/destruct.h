#ifndef ML_LIBRARY_MLP_DESTRUCT_H
#define ML_LIBRARY_MLP_DESTRUCT_H

#include "create.h"
#include <cstdlib>
#include <cstdio>

#include "MLP.h"

DLLEXPORT void destroy_mlp_prediction(float * prediction);
DLLEXPORT void destroy_mlp_model(MLP * mlp);
void destroy_mlp_W_model(MLP *mlp);
void destroy_mlp_deltas_model(MLP *mlp);
void destroy_mlp_X_model(MLP *mlp);



#endif //ML_LIBRARY_DESTRUCT_H
