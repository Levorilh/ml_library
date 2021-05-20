#include "../headers/linear/train.h"

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float *model,
                                                                 int input_dim,
                                                                 float *flattened_dataset_inputs,
                                                                 int samples_count,
                                                                 float *flattened_dataset_expected_outputs,
                                                                 float alpha,
                                                                 int iterations_count) {
    for (int it = 1; it <= iterations_count; it++) {
        int k = rand() % samples_count;
        float Yk = flattened_dataset_expected_outputs[k];
        float *Xk = new float[input_dim];
        for (int Xk_index = 0, flattened_dataset_inputs_index = k * input_dim;
             Xk_index < input_dim; Xk_index++, flattened_dataset_inputs_index++) {
            Xk[Xk_index] = flattened_dataset_inputs[flattened_dataset_inputs_index];
        }
        float gXk = predict_linear_model_classification(model, input_dim, Xk);
        model[0] += alpha * (Yk - gXk) * 1.0;    // bias correction
        for (int i = 1; i < (input_dim + 1) ; i++) {
            model[i] += alpha * (Yk - gXk) * Xk[i - 1];
        }
        //printf("Iteration %d  : [%f , %f  , %f ]\n" ,it, model[0] , model[1] , model[2]);
    }
}

DLLEXPORT void train_regression_pseudo_inverse_linear_model(float *model,
                                                            const int input_dim,
                                                            const float flattened_dataset_inputs[],
                                                            const int samples_count,
                                                            const float flattened_dataset_expected_outputs[]) {
    MatrixXd X(samples_count, input_dim + 1);
    for (int i = 0; i < samples_count; i++) {
        X(i, 0) = 1;
        for (int j = 1; j < input_dim + 1; j++) {
            X(i, j) = flattened_dataset_inputs[i * input_dim + j - 1];
        }
    }

    MatrixXd Y(samples_count, 1);
    for (int i = 0; i < samples_count; i++) {
        Y(i, 0) = flattened_dataset_expected_outputs[i];
    }

    MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;

    for (int i = 0; i < input_dim + 1; i++) {
        model[i] = W(i, 0);
    }
}