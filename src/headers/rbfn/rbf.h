//
// Created by ttres on 22/06/2021.
//

#ifndef ML_LIBRARY_RBF_H
#define ML_LIBRARY_RBF_H

#include <vector>
#include "Point.h"
#include <cmath>

#define MAX_DIM_GAP_ALLOWED 0.1
#define MAX_GAP_ALLOWED 0.5
Point** kmeans(float **X, int len_X,const int input_dim,const int k ,const int max_iters = 200);
Point** init_kmeans(const int cluster_count, float** dataset , const int dataset_size, const int input_dim);


#endif //ML_LIBRARY_RBF_H
