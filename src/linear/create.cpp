#include "../headers/linear/create.h"


DLLEXPORT float *create_linear_model(int input_dim) {
    auto *result = (float *)(malloc(sizeof(float) * (input_dim + 1)));
    for (int i = 0; i < input_dim + 1; i++) {
        result[i] = ((rand() % 2001) / 1000.0) - 1.;
    }
    return result;
}