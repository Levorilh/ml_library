//
// Created by Réda Maizate on 22/06/2021.
//

#include "../headers/tests/rbfn_model_test.h"

void test_get_distance_rbfn() {
    float x1[] = {3, 2, 3};
    int len_x1 = 3;
    float x2[] = {1, 2, 3};

    Point p1 = Point(1, x1, len_x1);
    Point p2 = Point(1, x2, len_x1);

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

    const int len_X = 11;
    const int input_dim = 2;
    const int k = 2;
    const int max_iters = 40;

    float **data = new float *[len_X];
    for (int i = 0; i < len_X; ++i) {
        data[i] = new float[input_dim];
    }
    data[0][0] = 2.9;
    data[0][1] = 5.6;
    data[1][0] = 7;
    data[1][1] = 5;
    data[2][0] = 8;
    data[2][1] = 4;
    data[3][0] = 4.6;
    data[3][1] = 9.3;

    data[4][0] = -1;
    data[4][1] = 2.3;
    data[5][0] = -3;
    data[5][1] = 0.5;
    data[6][0] = -2.5;
    data[6][1] = 0.3;
    data[7][0] = -3;
    data[7][1] = -2;
    data[8][0] = -1.5;
    data[8][1] = -1.9;
    data[9][0] = 2.3;
    data[9][1] = -1.5;
    data[10][0] = -3;
    data[10][1] = -2.5;

    Point **points = kmeans(data, len_X, input_dim, k, max_iters);

    cout << "middle 1 : " << points[0]->coords[0] << "  "
         << points[0]->coords[1] /*<< "  " << points[0]->coords[2] << "  " << points[0]->coords[3]*/ << "  " << endl;
    cout << "middle 2 : " << points[1]->coords[0] << "  "
         << points[1]->coords[1] /*<< "  " << points[1]->coords[2] << "  " << points[1]->coords[3]*/ << "  ";

     cout << "var 1 : " << points[0]->deviation << endl;
     cout << "var 2 : " << points[1]->deviation << endl;
//    delete[] data;
    delete[] points;
}