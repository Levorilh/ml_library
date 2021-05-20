#ifndef ML_LIBRARY_CREATE_H
#define ML_LIBRARY_CREATE_H

    #include <iostream>
    #include <cstdio>
    #include <cstdlib>
    #include <ctime>
    #include "../../../Eigen/Dense"

    #include "../main.h"

    #include "train.h"
    #include "predict.h"
    #include "destroy.h"

    using Eigen::MatrixXd;
    using namespace std;

    DLLEXPORT float *create_linear_model(int input_dim);

#endif //ML_LIBRARY_CREATE_H
