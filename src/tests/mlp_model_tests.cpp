//
// Created by Reda on 09/05/2021.
//

#include "../headers/main.h"


void test_regression_mlp() {
    int input_dim = 1;
    int total_input_dim = 3;

    float flattened_dataset_inputs[] = {
            -5,
            4,
            6,
    };

    float dataset_expected_outputs[] = {
            1.2,
            7,
            8.3
    };

    float input_test[] = {
            0.1, // 2.9563198900137584 (+/- 0.65)
            10.2, // 8.828858498298032 (+/- 0.5)
            -5.6 // 1.1826899086590916 (+/- 0.5)
    };

    int dims[total_input_dim];
    dims[0] = 1;
    dims[1] = 3;
    dims[2] = 1;

    MLP * model = create_mlp_model(dims, total_input_dim);

    cout << "#### TEST REGRESSION MLP ####" << endl;
    cout << "-- Before training --" << endl;

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];

        float * predicted_outputs = predict_mlp_model_regression(model,input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Expected output: " << dataset_expected_outputs[i] << endl;

        destroy_mlp_prediction(predicted_outputs);
    }

    train_regression_stochastic_gradient_backpropagation_mlp_model(model,
                                                                   flattened_dataset_inputs,
                                                                   (total_input_dim / input_dim),
                                                                   dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];

        float * predicted_outputs = predict_mlp_model_regression(model,input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Expected output: " << dataset_expected_outputs[i] << endl;

        destroy_mlp_prediction(predicted_outputs);
    }

    cout << "-- Test --" << endl;

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = input_test[i];

        float * predicted_outputs = predict_mlp_model_regression(model,input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Input: " << input_test[i] << endl;

        //cout << predicted_outputs[0] << " for :" << dataset_expected_outputs[i]  << endl;
        destroy_mlp_prediction(predicted_outputs);
    }

    cout << "####################################" << endl;

    destroy_mlp_model(model);
}

void test_classification_mlp() {
    int samples_count = 4;
    int input_dim = 2;
    float dataset_inputs[] = {
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    float dataset_expected_outputs[] = {
            -1,
            -1,
            1,
            1
    };

    float input_test[] = {
            /*
             * Ne pas oublier qu'on entraîne notre algorithme sur un exemple XOR, donc le pire exemple possible.
             * Du coup, faut que je continue a reflechir a des possibles exemples.
             */
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    int total_input_dim = 3;

    int dims[total_input_dim];
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 1;

    MLP *model = create_mlp_model(dims, total_input_dim);

    cout << "#### TEST CLASSIFICATION MLP ####" << endl;
    cout << "-- Before training --" << endl;

    for (int i = 0, j = 0; j < samples_count; i+=input_dim, j++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        float * predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / " << "Prediction: " << predicted_output[0] << " / Expected output: " << dataset_expected_outputs[j] << endl;

        destroy_mlp_prediction(predicted_output);
    }


    train_classification_stochastic_gradient_backpropagation_mlp_model(model,
                                                                       dataset_inputs,
                                                                       samples_count,
                                                                       dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for (int i = 0, j = 0; j < samples_count; i+=input_dim, j++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        float * predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / Prediction: " << predicted_output[0] << " / Expected output: " << dataset_expected_outputs[j] << endl;

        destroy_mlp_prediction(predicted_output);
    }

    cout << "-- Test --" << endl;

    for (int i = 0, j = 0; j < samples_count; i+=input_dim, j++) {
        float sub_input[2];
        sub_input[0] = input_test[i];
        sub_input[1] = input_test[i + 1];

        float * predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / Prediction: " << predicted_output[0] << endl;

        destroy_mlp_prediction(predicted_output);
    }

    cout << "####################################" << endl;

    destroy_mlp_model(model);
}

void test_multiclassification_mlp() {

    int dataset_inputs[] = {
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    int dataset_expected_outputs[] = {
            1, -1, -1,
            -1, 1, -1,
            -1, -1, 1
    };

    int total_input_dim = 3;

    int dims[3];
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 3;

    MLP *model = create_mlp_model(dims, total_input_dim);
}