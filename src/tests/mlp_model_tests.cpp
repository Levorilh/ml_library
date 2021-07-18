#include "../headers/tests/mlp_model_test.h"
#include "../headers/mlp/save.h"

void test_regression_mlp() {
    const int input_dim = 1;
    const int total_input_dim = 3;

    const float flattened_dataset_inputs[] = {
            -5,
            4,
            6,
    };

    const float dataset_expected_outputs[] = {
            1.2,
            7,
            8.3
    };

    const float input_test[] = {
            0.1, // 2.9563198900137584 (+/- 0.65)
            10.2, // 8.828858498298032 (+/- 0.5)
            -5.6 // 1.1826899086590916 (+/- 0.5)
    };

    int* dims = (int*)malloc(sizeof(int) * total_input_dim);
    dims[0] = 1;
    dims[1] = 3;
    dims[2] = 1;

    MLP* model = create_mlp_model(dims, total_input_dim);

    cout << "#### TEST REGRESSION MLP ####" << endl;
    cout << "-- Before training --" << endl;

    for (int i = 0; i < total_input_dim; i++) {
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];

        float *predicted_outputs = predict_mlp_model_regression(model, input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Expected output: " << dataset_expected_outputs[i] << endl;

        destroy_mlp_prediction(predicted_outputs);
    }

    train_regression_stochastic_gradient_backpropagation_mlp_model(model,
                                                                   flattened_dataset_inputs,
                                                                   (total_input_dim / input_dim),
                                                                   dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for (int i = 0; i < total_input_dim; i++) {
        float input_sub[1];
        input_sub[0] = flattened_dataset_inputs[i];

        float *predicted_outputs = predict_mlp_model_regression(model, input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Expected output: " << dataset_expected_outputs[i] << endl;

        destroy_mlp_prediction(predicted_outputs);
    }

    cout << "-- Test --" << endl;

    for (int i = 0; i < total_input_dim; i++) {
        float input_sub[1];
        input_sub[0] = input_test[i];

        float *predicted_outputs = predict_mlp_model_regression(model, input_sub);
        cout << "Prediction: " << predicted_outputs[0] << " / Input: " << input_test[i] << endl;

        //cout << predicted_outputs[0] << " for :" << dataset_expected_outputs[i]  << endl;
        destroy_mlp_prediction(predicted_outputs);
    }

    cout << "####################################" << endl;

    destroy_mlp_model(model);
}

void test_classification_mlp() {
    const int samples_count = 4;
    const int input_dim = 2;
    const float dataset_inputs[] = {
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    const float dataset_expected_outputs[] = {
            -1,
            -1,
            1,
            1
    };

    const float input_test[] = {
            /*
             * Ne pas oublier qu'on entraîne notre algorithme sur un exemple XOR, donc le pire exemple possible.
             * Du coup, faut que je continue a reflechir a des possibles exemples.
             */
            0, 0,
            1, 1,
            0, 1,
            1, 0
    };

    const int total_input_dim = 3;

    int * dims = (int *)malloc(sizeof(int) * total_input_dim);
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 1;

    MLP *model = create_mlp_model(dims, total_input_dim);

    cout << "#### TEST CLASSIFICATION MLP ####" << endl;
    cout << "-- Before training --" << endl;

    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        float *predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / " << "Prediction: "
             << predicted_output[0] << " / Expected output: " << dataset_expected_outputs[j] << endl;

        destroy_mlp_prediction(predicted_output);
    }


    train_classification_stochastic_gradient_backpropagation_mlp_model(model,
                                                                       dataset_inputs,
                                                                       samples_count,
                                                                       dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        float *predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / Prediction: " << predicted_output[0]
             << " / Expected output: " << dataset_expected_outputs[j] << endl;

        destroy_mlp_prediction(predicted_output);
    }

    cout << "-- Test --" << endl;

    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
        float sub_input[2];
        sub_input[0] = input_test[i];
        sub_input[1] = input_test[i + 1];

        float *predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / Prediction: " << predicted_output[0]
             << endl;

        destroy_mlp_prediction(predicted_output);
    }

    cout << "####################################" << endl;

    destroy_mlp_model(model);
}

void test_multiclassification_mlp() {
    const int samples_count = 3;

    const int input_dim = 2;

    const float dataset_inputs[] = {
            0., 0.,
            0.5, 0.5,
            1., 0.
    };

    const float dataset_expected_outputs[] = {
            1, -1, -1,
            -1, 1, -1,
            -1, -1, 1
    };


    int * dims = (int *)malloc(sizeof(int) * samples_count);
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 3;

    MLP *model = create_mlp_model(dims, samples_count);

    cout << "#### TEST MULTICLASSIFICATION MLP ####" << endl;
    cout << "-- Before training --" << endl;

    float *predicted_output;
    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
        float sub_input[2];
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        predicted_output = predict_mlp_model_classification(model, sub_input);
        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / " << "Prediction: ["
             << predicted_output[0] << ", " << predicted_output[1] << ", " << predicted_output[2] << "] / Expected output: [" << dataset_expected_outputs[j*3] << ", " << dataset_expected_outputs[j*3 + 1] << ", " << dataset_expected_outputs[j*3 + 2] << "]" << endl;

    }

//    destroy_mlp_prediction(predicted_output);
    train_classification_stochastic_gradient_backpropagation_mlp_model(model,
                                                                       dataset_inputs,
                                                                       samples_count,
                                                                       dataset_expected_outputs);

    cout << "-- After training --" << endl;

    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
        float * sub_input = (float *)malloc(sizeof(float) * input_dim);
        sub_input[0] = dataset_inputs[i];
        sub_input[1] = dataset_inputs[i + 1];

        predicted_output = predict_mlp_model_classification(model, sub_input);

        cout << "Input-checker: [" << sub_input[0] << "," << sub_input[1] << "] / " << "Prediction: ["
                << predicted_output[0] << ", " << predicted_output[1] << ", " << predicted_output[2] << "] / Expected output: [" << dataset_expected_outputs[j*3] << ", " << dataset_expected_outputs[j*3 + 1] << ", " << dataset_expected_outputs[j*3 + 2] << "]" << endl;

        free(sub_input);
    }
//    destroy_mlp_prediction(predicted_output);

//    cout << "-- Test --" << endl;
//
//    for (int i = 0, j = 0; j < samples_count; i += input_dim, j++) {
//        float dataset_test[2];
//        dataset_test[0] = input_test[i];
//        dataset_test[1] = input_test[i + 1];
//
//        float *predicted_output = predict_mlp_model_classification(model, dataset_test);
//
//        cout << "Input-checker: [" << dataset_test[0] << "," << dataset_test[1] << "] / Prediction: " << predicted_output[0]
//             << endl;
//
//        destroy_mlp_prediction(predicted_output);
//    }

    cout << "####################################" << endl;

    float dataset_test[2];
    dataset_test[0] = 100;
    dataset_test[1] = 0;
    predicted_output = predict_mlp_model_classification(model, dataset_test);

    cout << "Input-checker: [" << dataset_test[0] << "," << dataset_test[1] << "] / " << "Prediction: ["
         << predicted_output[0] << ", " << predicted_output[1] << ", " << predicted_output[2] << "]" << endl;

//    destroy_mlp_prediction(predicted_output);

    destroy_mlp_model(model);
}

void test_save_mlp_model(){
    //char *path = "C:\\Users\\N\\Desktop\\test_mlp_model.txt";
    char *path = "C:\\Users\\ttres\\Desktop\\3A\\S2\\MachineLearning\\ml_library\\test_mlp_model.txt";
//    const char* path = "/Users/redamaizate/Documents/3IABD/Projet-Annuel\\ml_library\\src\\tests\\tests_mlp_model.txt";
    //char* path = "test_mlp_save.txt";

    int total_input_dim = 3;
    int* dims = (int*)malloc(sizeof(int) * total_input_dim);
    dims[0] = 2;
    dims[1] = 3;
    dims[2] = 2;

    MLP* model = create_mlp_model(dims, total_input_dim);

    save_mlp_model(model, path);
    
    MLP* new_model = load_mlp_model(path);

    auto** X = (float**)malloc(sizeof(float*) * 2);
    X[0] = (float*)malloc(sizeof(float) * 2);
    X[1] = (float*)malloc(sizeof(float) * 2);

    X[0][0] = 1.f;
    X[0][1] = 1.f;

    X[1][0] = 1.f;
    X[1][1] = 1.f;

    float* pred = predict_mlp_model_classification(new_model , X[0] );

    cout << "res 1 : " << pred[0] << "  " << pred[1] << endl;

    free(pred);
    pred = predict_mlp_model_classification(new_model , X[1] );
    cout << "res 2 : " << pred[0] << "  " << pred[1] << endl;
    free(pred);


    pred = predict_mlp_model_classification(model , X[1] );
    cout << "res 3 : " << pred[0] << "  " << pred[1] << endl;
    free(pred);


    destroy_mlp_model(model);
    destroy_mlp_model(new_model);
    free(dims);
}

void test_img_mlp_model(){
    auto* img = new float[192];
    img[ 0 ] =  0.43  ;
    img[ 1 ] =  0.64  ;
    img[ 2 ] =  0.87  ;
    img[ 3 ] =  0.43  ;
    img[ 4 ] =  0.62  ;
    img[ 5 ] =  0.83  ;
    img[ 6 ] =  0.45  ;
    img[ 7 ] =  0.58  ;
    img[ 8 ] =  0.77  ;
    img[ 9 ] =  0.4  ;
    img[ 10 ] =  0.62  ;
    img[ 11 ] =  0.85  ;
    img[ 12 ] =  0.48  ;
    img[ 13 ] =  0.59  ;
    img[ 14 ] =  0.7  ;
    img[ 15 ] =  0.59  ;
    img[ 16 ] =  0.56  ;
    img[ 17 ] =  0.51  ;
    img[ 18 ] =  0.61  ;
    img[ 19 ] =  0.6  ;
    img[ 20 ] =  0.55  ;
    img[ 21 ] =  0.63  ;
    img[ 22 ] =  0.62  ;
    img[ 23 ] =  0.58  ;
    img[ 24 ] =  0.47  ;
    img[ 25 ] =  0.6  ;
    img[ 26 ] =  0.76  ;
    img[ 27 ] =  0.48  ;
    img[ 28 ] =  0.62  ;
    img[ 29 ] =  0.78  ;
    img[ 30 ] =  0.48  ;
    img[ 31 ] =  0.55  ;
    img[ 32 ] =  0.7  ;
    img[ 33 ] =  0.45  ;
    img[ 34 ] =  0.6  ;
    img[ 35 ] =  0.78  ;
    img[ 36 ] =  0.6  ;
    img[ 37 ] =  0.64  ;
    img[ 38 ] =  0.7  ;
    img[ 39 ] =  0.73  ;
    img[ 40 ] =  0.71  ;
    img[ 41 ] =  0.65  ;
    img[ 42 ] =  0.76  ;
    img[ 43 ] =  0.74  ;
    img[ 44 ] =  0.69  ;
    img[ 45 ] =  0.72  ;
    img[ 46 ] =  0.69  ;
    img[ 47 ] =  0.64  ;
    img[ 48 ] =  0.57  ;
    img[ 49 ] =  0.56  ;
    img[ 50 ] =  0.59  ;
    img[ 51 ] =  0.57  ;
    img[ 52 ] =  0.52  ;
    img[ 53 ] =  0.55  ;
    img[ 54 ] =  0.46  ;
    img[ 55 ] =  0.25  ;
    img[ 56 ] =  0.27  ;
    img[ 57 ] =  0.44  ;
    img[ 58 ] =  0.25  ;
    img[ 59 ] =  0.28  ;
    img[ 60 ] =  0.58  ;
    img[ 61 ] =  0.48  ;
    img[ 62 ] =  0.51  ;
    img[ 63 ] =  0.62  ;
    img[ 64 ] =  0.51  ;
    img[ 65 ] =  0.48  ;
    img[ 66 ] =  0.74  ;
    img[ 67 ] =  0.71  ;
    img[ 68 ] =  0.66  ;
    img[ 69 ] =  0.75  ;
    img[ 70 ] =  0.72  ;
    img[ 71 ] =  0.67  ;
    img[ 72 ] =  0.59  ;
    img[ 73 ] =  0.59  ;
    img[ 74 ] =  0.63  ;
    img[ 75 ] =  0.62  ;
    img[ 76 ] =  0.58  ;
    img[ 77 ] =  0.61  ;
    img[ 78 ] =  0.66  ;
    img[ 79 ] =  0.32  ;
    img[ 80 ] =  0.31  ;
    img[ 81 ] =  0.55  ;
    img[ 82 ] =  0.21  ;
    img[ 83 ] =  0.2  ;
    img[ 84 ] =  0.6  ;
    img[ 85 ] =  0.4  ;
    img[ 86 ] =  0.4  ;
    img[ 87 ] =  0.51  ;
    img[ 88 ] =  0.32  ;
    img[ 89 ] =  0.31  ;
    img[ 90 ] =  0.52  ;
    img[ 91 ] =  0.45  ;
    img[ 92 ] =  0.42  ;
    img[ 93 ] =  0.73  ;
    img[ 94 ] =  0.68  ;
    img[ 95 ] =  0.63  ;
    img[ 96 ] =  0.6  ;
    img[ 97 ] =  0.53  ;
    img[ 98 ] =  0.55  ;
    img[ 99 ] =  0.64  ;
    img[ 100 ] =  0.51  ;
    img[ 101 ] =  0.51  ;
    img[ 102 ] =  0.78  ;
    img[ 103 ] =  0.45  ;
    img[ 104 ] =  0.4  ;
    img[ 105 ] =  0.64  ;
    img[ 106 ] =  0.38  ;
    img[ 107 ] =  0.33  ;
    img[ 108 ] =  0.62  ;
    img[ 109 ] =  0.31  ;
    img[ 110 ] =  0.31  ;
    img[ 111 ] =  0.58  ;
    img[ 112 ] =  0.29  ;
    img[ 113 ] =  0.28  ;
    img[ 114 ] =  0.43  ;
    img[ 115 ] =  0.22  ;
    img[ 116 ] =  0.19  ;
    img[ 117 ] =  0.47  ;
    img[ 118 ] =  0.35  ;
    img[ 119 ] =  0.31  ;
    img[ 120 ] =  0.45  ;
    img[ 121 ] =  0.4  ;
    img[ 122 ] =  0.38  ;
    img[ 123 ] =  0.56  ;
    img[ 124 ] =  0.49  ;
    img[ 125 ] =  0.47  ;
    img[ 126 ] =  0.63  ;
    img[ 127 ] =  0.61  ;
    img[ 128 ] =  0.58  ;
    img[ 129 ] =  0.73  ;
    img[ 130 ] =  0.61  ;
    img[ 131 ] =  0.59  ;
    img[ 132 ] =  0.47  ;
    img[ 133 ] =  0.29  ;
    img[ 134 ] =  0.27  ;
    img[ 135 ] =  0.14  ;
    img[ 136 ] =  0.09  ;
    img[ 137 ] =  0.07  ;
    img[ 138 ] =  0.11  ;
    img[ 139 ] =  0.07  ;
    img[ 140 ] =  0.06  ;
    img[ 141 ] =  0.25  ;
    img[ 142 ] =  0.2  ;
    img[ 143 ] =  0.18  ;
    img[ 144 ] =  0.51  ;
    img[ 145 ] =  0.47  ;
    img[ 146 ] =  0.47  ;
    img[ 147 ] =  0.64  ;
    img[ 148 ] =  0.57  ;
    img[ 149 ] =  0.58  ;
    img[ 150 ] =  0.64  ;
    img[ 151 ] =  0.64  ;
    img[ 152 ] =  0.65  ;
    img[ 153 ] =  0.67  ;
    img[ 154 ] =  0.58  ;
    img[ 155 ] =  0.58  ;
    img[ 156 ] =  0.5  ;
    img[ 157 ] =  0.34  ;
    img[ 158 ] =  0.31  ;
    img[ 159 ] =  0.06  ;
    img[ 160 ] =  0.07  ;
    img[ 161 ] =  0.04  ;
    img[ 162 ] =  0.05  ;
    img[ 163 ] =  0.05  ;
    img[ 164 ] =  0.04  ;
    img[ 165 ] =  0.27  ;
    img[ 166 ] =  0.19  ;
    img[ 167 ] =  0.17  ;
    img[ 168 ] =  0.4  ;
    img[ 169 ] =  0.38  ;
    img[ 170 ] =  0.39  ;
    img[ 171 ] =  0.36  ;
    img[ 172 ] =  0.38  ;
    img[ 173 ] =  0.36  ;
    img[ 174 ] =  0.27  ;
    img[ 175 ] =  0.27  ;
    img[ 176 ] =  0.25  ;
    img[ 177 ] =  0.22  ;
    img[ 178 ] =  0.23  ;
    img[ 179 ] =  0.2  ;
    img[ 180 ] =  0.46  ;
    img[ 181 ] =  0.32  ;
    img[ 182 ] =  0.29  ;
    img[ 183 ] =  0.28  ;
    img[ 184 ] =  0.25  ;
    img[ 185 ] =  0.24  ;
    img[ 186 ] =  0.29  ;
    img[ 187 ] =  0.25  ;
    img[ 188 ] =  0.23  ;
    img[ 189 ] =  0.42  ;
    img[ 190 ] =  0.33  ;
    img[ 191 ] =  0.32  ;

    int* layers = new int[2];
    layers[0] = 192;
    layers[1] = 8;

    MLP* p_model = create_mlp_model(layers , 2);
    printf(" oui !!%d" , p_model->d[p_model->d_length - 1]);
//    auto pred = new float[p_model->d_length - 1];
    float* pred = predict_mlp_model_classification(p_model , img);
//    delete[] pred;


//    printf("%s\n" , typeid(pred).name());
    printf("%f\n" , pred[0]);
//    pred = predict_mlp_model_classification(p_model , img);
//    auto* pred = new float[10];


    destroy_mlp_model(p_model);
    destroy_mlp_prediction(pred);

    delete[] img;
    delete[] layers;
}


void test_load_mlp_model(){
    MLP* model = load_mlp_model(R"(C:\Users\ttres\Desktop\3A\S2\MachineLearning\ml_library\please.txt)");

    auto* data = (float*)malloc(sizeof(int) * model->d[0]);
    for(int i = 0 ; i < model->d[0] ; i+=1){
        data[i] = 0.25f;
    }

    float* pred = predict_mlp_model_classification(model, data );
    printf("%d %lf //// %d  %lf\n", 0 , pred[0] , 1 , pred[1]);
    destroy_mlp_prediction(pred);
    free(data);
    destroy_mlp_model(model);
}