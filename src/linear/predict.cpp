#include "../headers/linear/predict.h"




DLLEXPORT float predict_linear_model_regression(float *model, int model_length, float *sample_inputs) {
    float result = model[0] * 1.0;    // bias
    for (int i = 1; i < model_length + 1; i++) {
        result += model[i] * sample_inputs[i - 1];
    }
    return result;
}

DLLEXPORT float predict_linear_model_classification(float *model, int model_length, float *sample_inputs) {
    if (predict_linear_model_regression(model, model_length, sample_inputs) >= 0) {
        return 1.0;
    } else {
        return -1.0;
    }
}