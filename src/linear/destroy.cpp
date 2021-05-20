#include "../headers/linear/destroy.h"

DLLEXPORT void destroy_linear_model(float *model) {
    free(model);
}