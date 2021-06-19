#include "../headers/tests/linear_model_test.h"
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
            4, 7,
    };

    float input_test[] = {
            5, 5, // 1
            10, 5, // 1
            0, 0, // 1 ou -1 (ça varie entre les deux, c'est normal)
    };

    cout << "#### TEST CLASSIFICATION LINEAR ####" << endl;
    cout << "-- Before training --" << endl;

    for(int i = 0, j = 0; i < total_input_dim; i += input_dim, j++) {
        float input_sub[2] ;
        input_sub[0]  = input[i];
        input_sub[1]  = input[i+1];

        float predict_before_train = predict_linear_model_classification(model, input_dim, input_sub);
        cout << "Prediction: " << predict_before_train << " / Expected output: " << out_put[j] << endl;
    }

    cout << "-- After training --" << endl;

    train_classification_rosenblatt_rule_linear_model(model,
                                                      input_dim,
                                                      input,
                                                      (total_input_dim / input_dim),
                                                      out_put);

    for(int i=0, j=0; i<total_input_dim; i+=input_dim, j++) {
        float input_sub[2] ;
        input_sub[0]  = input[i];
        input_sub[1]  = input[i+1];

        float predict_after_train = predict_linear_model_classification(model, input_dim, input_sub);
        cout << "Prediction: " << predict_after_train << " / Expected output: " << out_put[j] << endl;
    }

    cout << "-- Test --" << endl;


    for(int i=0, j=0; i < total_input_dim; i += input_dim, j++) {
        float input_sub[2] ;
        input_sub[0]  = input_test[i];
        input_sub[1]  = input_test[i+1];

        float predict_test = predict_linear_model_classification(model, input_dim, input_sub);
        cout << "Prediction: " << predict_test << " / Inputs: [" << input_sub[0] << "," << input_sub[1] << "]" << endl;
    }

    cout << "####################################" << endl;

    save_linear_model(model, input_dim, "save.txt");

    destroy_linear_model(model);
}




void test_regression_linear(){
    const int input_dim = 1;
    const int total_input_dim = 3;

    const float flattened_dataset_inputs[] = {
            -1,
            4,
            6,
    };

    const float dataset_expected_outputs[] = {
            5.2,
            7,
            8.3
    };

    const float input_test[] = {
            0.1, // 6.431019417475729
            10.2, // 9.204417475728157
            -5.6 // 4.967281553398059
    };


    float * model = create_linear_model(input_dim);

    cout << "#### TEST REGRESSION LINEAR ####" << endl;
    cout << "-- Before training --" << endl;

    auto * predicted_outputs = (float *)malloc(sizeof(float) * total_input_dim);

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];
        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        cout << "Prediction: " << predicted_outputs[i] << " / Expected output: " << dataset_expected_outputs[i] << endl;
    }


    train_regression_pseudo_inverse_linear_model(model,
                                                 input_dim,
                                                 flattened_dataset_inputs,
                                                 (total_input_dim / input_dim),
                                                 dataset_expected_outputs);


    cout << "-- After training --" << endl;

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];

        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        cout << "Prediction: " << predicted_outputs[i] << " / Expected output: " << dataset_expected_outputs[i] << endl;
    }

    cout << "-- Test --" << endl;

    for(int i = 0; i < total_input_dim; i++){
        float input_sub[1];
        input_sub[0] = input_test[i];

        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        cout << "Prediction: " << predicted_outputs[i] << " / Input: " << input_test[i] << endl;
    }

    cout << "####################################" << endl;

    destroy_linear_model(model);
}


void test_regression_linear_tricky(){
    const int input_dim = 2;
    const int total_input_dim = 8;

    const float flattened_dataset_inputs[] = {
            1, 1,
            2, 2,
            3, 3,
            3, 3,
    };

    const float dataset_expected_outputs[] = {
            1,
            2,
            3,
            3,
    };

    float * model = create_linear_model(input_dim);

    cout << "#### TEST REGRESSION LINEAR ####" << endl;
    cout << "-- Before training --" << endl;

    auto * predicted_outputs = (float *)malloc(sizeof(float) * total_input_dim);

    for(int i = 0; i < total_input_dim; i += 2){
        float input_sub[2];
        input_sub[0] = flattened_dataset_inputs[i];
        input_sub[1] = flattened_dataset_inputs[i + 1];
        //input_sub[2] = flattened_dataset_inputs[i + 2];
        //input_sub[3] = flattened_dataset_inputs[i + 3];
        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        cout << "Prediction: " << predicted_outputs[i] << " / Expected output: " << dataset_expected_outputs[i/input_dim] << endl;
    }


    train_regression_pseudo_inverse_linear_model(model,
                                                 input_dim,
                                                 flattened_dataset_inputs,
                                                 (total_input_dim / input_dim),
                                                 dataset_expected_outputs);


    cout << "-- After training --" << endl;

    for(int i = 0; i < total_input_dim; i += 2){
        float input_sub[2];
        input_sub[0] = flattened_dataset_inputs[i];
        input_sub[1] = flattened_dataset_inputs[i + 1];
        //input_sub[2] = flattened_dataset_inputs[i + 2];
        //input_sub[3] = flattened_dataset_inputs[i + 3];

        predicted_outputs[i] = predict_linear_model_regression(model,input_dim,input_sub);
        cout << "Prediction: " << predicted_outputs[i] << " / Expected output: " << dataset_expected_outputs[i/input_dim] << endl;
    }

    cout << "-- Test --" << endl;


    cout << "####################################" << endl;

    destroy_linear_model(model);
}

void test_load_linear(){
    int * input_dim = (int*)malloc(sizeof(int));
    float * model = load_linear_model("C:\\Users\\N\\Desktop\\test_linear_model.txt", input_dim);
    for(int i =0; i< (*input_dim) +1;i++){
        cout<<model[i]<<endl;
    }
    free(input_dim);
    free(model);
}