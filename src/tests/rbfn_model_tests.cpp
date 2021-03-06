//
// Created by Réda Maizate on 22/06/2021.
//

#include "../headers/tests/rbfn_model_test.h"

void test_save_load_rbf(){
    int input_dim = 2;
    int samples_count = 4;
    int num_classes = 2;
    int k = 4;
    bool naif = true;

    double * flatenned_input = (double*)malloc(sizeof(double)*samples_count*input_dim);
    double input[] =  {0., 0., 1., 1., 0., 1., 1., 0.};
    for(int i = 0; i < samples_count*input_dim; i++){
        flatenned_input[i] = input[i];
    }
    double * flatenned_output = (double*)malloc(sizeof(double)*samples_count*num_classes);
    double output[] =  {1., 0., 1., 0., 0., 1., 0., 1.};
    for(int i = 0; i < samples_count*num_classes; i++){
        flatenned_output[i] = output[i];
    }
    double * test_input = (double*)malloc(sizeof(double)*input_dim);
    double tinput[] = {1., 1.};
    for(int i = 0; i < input_dim; i++){
        test_input[i] = tinput[i];
    }
    double * test_output_expected = (double*)malloc(sizeof(double)*num_classes);
    double toutput[] = {1., 0.};
    for(int i = 0; i < num_classes; i++){
        test_output_expected[i] = toutput[i];
    }

    RBF* model = create_rbfn_model(input_dim,num_classes, k);
    train_rbfn_model(model,flatenned_input,samples_count,flatenned_output,naif);
    double * rslt1 = predict_rbfn(model,test_input);

    char* path = "C:\\Users\\N\\Desktop\\test_rbf_model.txt";
    save_rbf_model(model, path);
    destroy_rbfn_model(model);

    model = load_rbf_model(path);
    double * rslt2 = predict_rbfn(model,test_input);

    for(int i = 0; i < num_classes; i++){
        cout << "conpare rslt predict:" << rslt1[i] << " : " << rslt2[i] << endl;
    }

    free(flatenned_input);
    free(flatenned_output);
    free(test_input);
    free(test_output_expected);
}

void test_rbfn(){
    int input_dim = 2;
    int samples_count = 3;
    int num_classes = 3;
    int k = 3;
    bool naif = true;

    double * flatenned_input = (double*)malloc(sizeof(double)*samples_count*input_dim);
    double input[] =  {0.,0.,1.,1.,2.,0.};
    for(int i = 0; i < samples_count*input_dim; i++){
        flatenned_input[i] = input[i];
    }
    double * flatenned_output = (double*)malloc(sizeof(double)*samples_count*num_classes);
    double output[] =  //{1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.};
            {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    for(int i = 0; i < samples_count*num_classes; i++){
        flatenned_output[i] = output[i];
    }
    double * test_input = (double*)malloc(sizeof(double)*input_dim);
    double tinput[] = //{-5., 4.};
            {0., 0.};
    for(int i = 0; i < input_dim; i++){
        test_input[i] = tinput[i];
    }
    double * test_output_expected = (double*)malloc(sizeof(double)*num_classes);
    double toutput[] = //{1., 0.};
            {1., 0., 0.};
    for(int i = 0; i < num_classes; i++){
        test_output_expected[i] = toutput[i];
    }

    RBF* model = create_rbfn_model(input_dim,num_classes, k);
    train_rbfn_model(model,flatenned_input,samples_count,flatenned_output,naif);
    double * rslt = predict_rbfn(model,test_input);

    for(int i = 0 ; i < num_classes; i++){
        cout << "expected ["<<i<<"] = " << toutput[i] << endl;
    }


    free(rslt);
    free(flatenned_input);
    free(flatenned_output);
    free(test_input);
    free(test_output_expected);
}

void test_rbfn_XOR(){
    int input_dim = 2;
    int samples_count = 4;
    int num_classes = 2;
    int k = 4;
    bool naif = true;

    double * flatenned_input = (double*)malloc(sizeof(double)*samples_count*input_dim);
    double input[] =  //{-3., 3., -3., 2., -4., 2., -4., 3., -3.5, 3.5, 4., 1., 5., 1., 4., 0.5, 5., 0.5, 4.5, 1.5};
            {0., 0., 1., 1., 0., 1., 1., 0.};
    for(int i = 0; i < samples_count*input_dim; i++){
        flatenned_input[i] = input[i];
    }
    double * flatenned_output = (double*)malloc(sizeof(double)*samples_count*num_classes);
    double output[] =  //{1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.};
            {1., 0., 1., 0., 0., 1., 0., 1.};
    for(int i = 0; i < samples_count*num_classes; i++){
        flatenned_output[i] = output[i];
    }
    double * test_input = (double*)malloc(sizeof(double)*input_dim);
    double tinput[] = //{-5., 4.};
            {0., 1.};
    for(int i = 0; i < input_dim; i++){
        test_input[i] = tinput[i];
    }
    double * test_output_expected = (double*)malloc(sizeof(double)*num_classes);
    double toutput[] = //{1., 0.};
            {0., 1.};
    for(int i = 0; i < num_classes; i++){
        test_output_expected[i] = toutput[i];
    }

    RBF* model = create_rbfn_model(input_dim,num_classes, k);
    train_rbfn_model(model,flatenned_input,samples_count,flatenned_output,naif);
    double * rslt = predict_rbfn(model,test_input);

    cout << "expected [0] = " << toutput[0] << endl;
    cout << "expected [1] = " << toutput[1] << endl;

    free(rslt);
    free(flatenned_input);
    free(flatenned_output);
    free(test_input);
    free(test_output_expected);
}

void test_get_distance_rbfn() {
    double x1[] = {3, 2, 3};
    int len_x1 = 3;
    double x2[] = {1, 2, 3};

    Centroid p1 = Centroid(1, x1, len_x1);
    Centroid p2 = Centroid(1, x2, len_x1);

    double res = p1.distance_to(&p2);
    cout << res << endl;
    res = p2.distance_to(&p1);
    cout << res << endl;
}

void test_kmeans_rbfn() {
//    float X[2][2] = {
//            {1.f, 2.f},
//            {3.f, 4.f}
//    };

//    int len_X = 30;
//    int input_dim = 2;
//    int k = 2;
//    int max_iters = 40;
//
//    auto X = new float*[len_X];
//    for (int i = 0; i < len_X; ++i) {
//        X[i]= new float[input_dim];
//        cout << "point " << i  << "  ";
//        for(int j = 0; j < input_dim ; j += 1){
//            X[i][j] = (float)((rand() % 40)-20);
//            cout << X[i][j]  << "  ";
//        }
//        cout << endl;
//    }

    const int len_X = 14;
    const int input_dim = 2;
    const int k = 3;
    const int max_iters = 100;

    auto **data = new double *[len_X];
    for (int i = 0; i < len_X; ++i) {
        data[i] = new double[input_dim];
    }
    data[0][0] = -3.;
    data[0][1] = 3.;

    data[1][0] = -3.;
    data[1][1] = 2.;

    data[2][0] = -4.;
    data[2][1] = 3.;

    data[3][0] = -4.;
    data[3][1] = 2.;


    data[4][0] = 4.;
    data[4][1] = 1.;

    data[5][0] = 5.;
    data[5][1] = 1.;

    data[6][0] = 4;
    data[6][1] = 0.5;

    data[7][0] = 5;
    data[7][1] = 0.5;

    data[8][0] = -3.5;
    data[8][1] = 3.5;

    data[9][0] = 4.5;
    data[9][1] = 1.5;

    data[10][0] = 1.;
    data[10][1] = 4.5;

    data[11][0] = 1.;
    data[11][1] = 4.;


    data[12][0] = 1.5;
    data[12][1] = 4.5;

    data[13][0] = 1.5;
    data[13][1] = 4.;


//    data[8][0] = -1.5;
//    data[8][1] = -1.9;
//    data[9][0] = 2.3;
//    data[9][1] = -1.5;
//    data[10][0] = -3;
//    data[10][1] = -2.5;

//    for(int i = 0 ; i < 10 ; i += 1) {
    auto *ct1 = new Centroid(1, data[2], 2);
//todo make sur all tool functions work
    Centroid **points = train_kmeans(data, len_X, input_dim, k, max_iters);

//    cout << "middle 1 : " << points[0]->coords[0] << "  "
//         << points[0]->coords[1] /*<< "  " << points[0]->coords[2] << "  " << points[0]->coords[3]*/ << "  "
//         << endl;
//    cout << "middle 2 : " << points[1]->coords[0] << "  "
//         << points[1]->coords[1] /*<< "  " << points[1]->coords[2] << "  " << points[1]->coords[3]*/ << "  "
//         << endl;
//
//    cout << "middle 3 : " << points[2]->coords[0] << "  "
//         << points[2]->coords[1] /*<< "  " << points[1]->coords[2] << "  " << points[1]->coords[3]*/ << "  "
//         << endl;


//        cout << "var 1 : " << points[0]->deviation << endl;
//        cout << "var 2 : " << points[1]->deviation << endl;
    //    delete[] data;
    delete[] points;
//    }
}