#ifndef ML_LIBRARY_PREDICT_H
#define ML_LIBRARY_PREDICT_H


#include "../main.h"

DLLEXPORT float predict_linear_model_regression(float *model, int model_length, float *sample_inputs);
DLLEXPORT float predict_linear_model_classification(float *model, int model_length, float *sample_inputs);

#endif //ML_LIBRARY_PREDICT_H
