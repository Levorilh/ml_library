#ifndef ML_LIBRARY_MLP_H
#define ML_LIBRARY_MLP_H

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../../../Eigen/Dense"

using Eigen::MatrixXd;
using namespace std;

#include "../main.h"
#include "create.h"

#include "destruct.h"
#include "predict.h"

#include "train.h"


void forward_pass(MLP* mlp, const float sample_inputs[], const bool is_classification);

#endif //ML_LIBRARY_MLP_H
