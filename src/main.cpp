#include "headers/main.h"


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

    int nb_couches = 3;
    int dims[nb_couches];
    dims[0] = 1;
    dims[1] = 3;
    dims[2] = 1;

    MLP * model = create_mlp_model(dims, nb_couches);

    cout << "Before training:" << endl;
    //float predicted_outputs[total_input_dim];
    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];
        float * predicted_outputs = predict_mlp_model_regression(model,input_sub);
        cout << predicted_outputs[0] << endl;
        destroy_mlp_prediction(predicted_outputs);
    }

    train_regression_stochastic_gradient_backpropagation_mlp_model(model,
                                                                   flattened_dataset_inputs,
                                                                   (total_input_dim / input_dim),
                                                                   dataset_expected_outputs);

    cout << "After training:" << endl;
    for(int i =0; i < 3; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];
        float * predicted_outputs = predict_mlp_model_regression(model,input_sub);
        cout << predicted_outputs[0] << " for :" << dataset_expected_outputs[i]  << endl;
        destroy_mlp_prediction(predicted_outputs);
    }

    destroy_mlp_model(model);
}

void test_classification_mlp() {

    /* Classification à 3 classes

    int * dataset_inputs = {
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    int * dataset_expected_outputs = {
            1, -1, -1,
            -1, 1, -1,
            -1, -1, 1
    };

    MLP * model = create_mlp_model({2, 3, 3});
    */

    int samples_count = 4;
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

    int dims[3];
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 1;

    MLP *model = create_mlp_model(dims, 3);

    cout << "-- Before training --" << endl;

    for (int i = 0; i < samples_count; i++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];
        float *predicted_output = predict_mlp_model_classification(model, sub_input);
        cout << "Inputs : [" << sub_input[0] << "," << sub_input[1] << "] -> Prediction : " << predicted_output[0]
             << endl;
        destroy_mlp_prediction(predicted_output);
    }


    train_classification_stochastic_gradient_backpropagation_mlp_model(model,
                                                                       dataset_inputs,
                                                                       samples_count,
                                                                       dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for (int i = 0; i < samples_count; i++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];
        float *predicted_output = predict_mlp_model_classification(model, sub_input);
        cout << "Inputs : [" << sub_input[0] << "," << sub_input[1] << "] -> Prediction : " << predicted_output[0]
             << endl;
        destroy_mlp_prediction(predicted_output);
    }

    destroy_mlp_model(model);
}

void test_classification_linear(){
    int input_dim = 2;

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

        float predict_before_train = predict_linear_model_classification(model, input_dim, input_sub);
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

        float predict_before_train = predict_linear_model_classification(model, input_dim, input_sub);
        std::cout << predict_before_train << std::endl;
    }

    destroy_linear_model(model);
}

void test_regression_linear(){
    int input_dim = 1;
    int total_input_dim = 3;

    float flattened_dataset_inputs[] = {
            -5,
            4,
            6,
    };

    float dataset_expected_outputs[] = {
            5.2,
            7,
            8.3
    };


    float * model = create_linear_model(input_dim);

    std::cout << "Before training:" << std::endl;
    float predicted_outputs[total_input_dim];
    for(int i =0; i<total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = dataset_expected_outputs[i];
        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        std::cout << predicted_outputs[i] << std::endl;
    }


    train_regression_pseudo_inverse_linear_model(model, input_dim, flattened_dataset_inputs, (total_input_dim / input_dim), dataset_expected_outputs);


    std::cout << "After training:" << std::endl;
    for(int i =0; i<3; i++){
        float input_sub[1];
        input_sub[0] = dataset_expected_outputs[i];
        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        std::cout <<  predicted_outputs[i] << " for :" << dataset_expected_outputs[i]  << std::endl;
    }

    destroy_linear_model(model);
}


int main(int argc, char** argv) {
    srand(time(nullptr));

    test_regression_mlp();

    return 0;
}