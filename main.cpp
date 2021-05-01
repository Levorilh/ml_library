#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "Eigen/Dense"
#include <experimental/random>

using Eigen::MatrixXd;

float *create_linear_model(int);

void destroy_linear_model(float *);

void train_regression_pseudo_inverse_linear_model(float *model,
                                                  int input_dim,
                                                  float *flattened_dataset_inputs,
                                                  int samples_count,
                                                  float *flattened_dataset_expected_outputs);

void train_classification_rosenblatt_rule_linear_model(float *model,
                                                       int input_dim,
                                                       float *flattened_dataset_inputs,
                                                       int samples_count,
                                                       float *flattened_dataset_expected_outputs,
                                                       float alpha = 0.001,
                                                       int iterations_count = 1000);

float predict_linear_model_regression(float *model, int model_length, float *sample_inputs);

float predict_linear_model_classification(float *model, int model_length, float *sample_inputs);


int main() {
//    std::cout << "Hello, World!" << std::end
    srand(time(nullptr));

    int input_dim = 2;
    int model_length = input_dim+1;

    float *model = create_linear_model(input_dim);

    float out_put[] = {
            1,
            1,
            -1
    };

    int total_input_dim = 6;
    float input[] = {
            3, 4,
            6, 5,
            4, 7
    };

    std::cout << "Before training:" << std::endl;

    for(int i=0; i<total_input_dim; i+=input_dim) {
        float input_sub[2] ;
        input_sub[0]  = input[i];
        input_sub[1]  = input[i+1];

        float predict_before_train = predict_linear_model_classification(model, model_length, input_sub);
        std::cout << predict_before_train << std::endl;
    }

    std::cout << "After training:" << std::endl;

    train_classification_rosenblatt_rule_linear_model(model,
                                                      input_dim,
                                                      input,
                                                      (total_input_dim / input_dim),
                                                      out_put);

    for(int i=0; i<total_input_dim; i+=input_dim) {
        float input_sub[2] ;
        input_sub[0]  = input[i];
        input_sub[1]  = input[i+1];

        float predict_before_train = predict_linear_model_classification(model, model_length, input_sub);
        std::cout << predict_before_train << std::endl;
    }

    destroy_linear_model(model);
    return 0;
}

float *create_linear_model(int input_dim) {
    auto *result = (float *) malloc(sizeof(float) * input_dim);
    for (int i = 0; i < input_dim; i++) {
        result[i] = ((rand() % 2001) / 1000.0) - 1.;
    }
    return result;
}

void destroy_linear_model(float *model) {
    free(model);
}

void train_classification_rosenblatt_rule_linear_model(float *model,
                                                       int input_dim,
                                                       float *flattened_dataset_inputs,
                                                       int samples_count,
                                                       float *flattened_dataset_expected_outputs,
                                                       float alpha,
                                                       int iterations_count) {
    for (int it = 0; it < iterations_count; it++) {
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

        //std::cout << "iter : " << it << std::endl;
        //for( int iter = 0 ; iter < input_dim+1 ; iter++){
        //    std::cout << model[iter] << std::endl;
        //}
        //std::cout << std::endl;

    }
}

void train_regression_pseudo_inverse_linear_model(float *model,
                                                  int input_dim,
                                                  float *flattened_dataset_inputs,
                                                  int samples_count,
                                                  float *flattened_dataset_expected_outputs) {
    MatrixXd X(samples_count, input_dim + 1);
    for (int i = 0; i < samples_count; i++) {
        X(i, 0) = 1;
        for (int j = 1; j < input_dim + 1; j++) {
            X(i, j) = flattened_dataset_inputs[i * samples_count + j];
        }
    }

    MatrixXd Y(samples_count, 1);
    for (int i = 0; i < samples_count; i++) {
        Y(i, 0) = flattened_dataset_expected_outputs[i];
    }

    Eigen:
    MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;

    for (int i = 0; i < input_dim; i++) {
        model[i] = W(i, 0);
    }
}


float predict_linear_model_regression(float *model, int model_length, float *sample_inputs) {
    float result = model[0] * 1.0;    // bias
    for (int i = 1; i < model_length; i++) {
        result += model[i] * sample_inputs[i - 1];
    }
    return result;
}

float predict_linear_model_classification(float *model, int model_length, float *sample_inputs) {
    if (predict_linear_model_regression(model, model_length, sample_inputs) >= 0) {
        return 1.0;
    } else {
        return -1.0;
    }
}

