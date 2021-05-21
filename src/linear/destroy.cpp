#include "../headers/linear/destruct.h"

DLLEXPORT void destroy_linear_model(float *model) {
    free(model);
}