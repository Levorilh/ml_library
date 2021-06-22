//
// Created by Réda Maizate on 22/06/2021.
//

#include "../headers/tests/rbfn_model_test.h"

void test_get_distance_rbfn() {
    float x1[] = {3, 2, 3};
    int len_x1 = 3;
    float x2[] = {1, 2, 3};

    double res = get_distance(x1, len_x1, x2);
    cout << res << endl;
}