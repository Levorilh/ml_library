#ifndef ML_LIBRARY_MLP_CREATE_H
#define ML_LIBRARY_MLP_CREATE_H

#include <cstdio>
#include <cstdlib>
#include <iostream>

class MLP_t{
public:
    int *d;
    float** X;
    float** deltas;
    float*** W;
    int d_length;
};
typedef class MLP_t MLP;


DLLEXPORT MLP* create_mlp_model(int* npl,const int npl_length);

#endif //ML_LIBRARY_CREATE_H
