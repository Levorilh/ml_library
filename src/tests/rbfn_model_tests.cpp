//
// Created by Réda Maizate on 22/06/2021.
//

#include "../headers/tests/rbfn_model_test.h"

void test_get_distance_rbfn() {
    float x1[] = {3, 2, 3};
    int len_x1 = 3;
    float x2[] = {1, 2, 3};

    Point p1 = Point(1, x1 , len_x1);
    Point p2 = Point(1, x2 , len_x1);

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

    int len_X = 15;
    int input_dim = 4;
    int k = 2;
    int max_iters = 3;

    auto X = new float*[len_X];
    for (int i = 0; i < len_X; ++i) {
        X[i]= new float[input_dim];
        for(int j = 0; j < input_dim ; j += 1){
            X[i][j] = (float)(rand() % 12);
        }
    }

    Point** points = kmeans(X, len_X, input_dim,k,max_iters);

    delete[] X;
    delete[] points;
}