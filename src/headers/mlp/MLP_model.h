#ifndef LIBRARY_MLP_MODEL_H
#define LIBRARY_MLP_MODEL_H



#include <cstdio>
    #include <cstdlib>
#include <cmath>

    #include "../../../Eigen/Dense"

    #include "../main.h"
    #include "create.h"

#include "MLP.h"
#include "destruct.h"
#include "predict.h"

    #include "train.h"

    using Eigen::MatrixXd;
    using namespace std;

    void forward_pass(MLP* mlp, const float sample_inputs[], const bool is_classification[]);
#endif //LIBRARY_MLP_MODEL_H

