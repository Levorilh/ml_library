//
// Created by ttres on 22/06/2021.
//

#ifndef ML_LIBRARY_RBF_H
#define ML_LIBRARY_RBF_H
#include <vector>

Point** kmeans(float **X, int len_X,const int input_dim,int k , int max_iters = 200);
Point** init_kmeans(const int cluster_count, float** dataset , const int dataset_size, const int input_dim);


#endif //ML_LIBRARY_RBF_H
