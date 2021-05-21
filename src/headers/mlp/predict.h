#ifndef ML_LIBRARY_PREDICT_H
#define ML_LIBRARY_PREDICT_H


DLLEXPORT float * predict_mlp_model_regression(MLP* mlp, float* sample_input);
DLLEXPORT float * predict_mlp_model_classification(MLP* mlp, float* sample_input);

#endif //ML_LIBRARY_PREDICT_H
