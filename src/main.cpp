#include "headers/main.h"

int main(int argc, char** argv) {

    printf("yo");

    srand(time(nullptr));

    // TEST REGRESSION
/*
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

    return 0;
*/
    // TEST CLASSIFICATION
/*
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
    return 0;
*/
    return 0;
}

