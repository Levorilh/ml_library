#ifndef ML_LIBRARY_CREATE_H
#define ML_LIBRARY_CREATE_H

    #include "../main.h"
    struct MLP_t{
        int d_length;
        int *d;
        float** X;
        float** deltas;
        float*** W;
    };

    typedef struct MLP_t MLP;

    DLLEXPORT MLP* create_mlp_model(int* npl, const int npl_length);

#endif //ML_LIBRARY_CREATE_H
