//
// Created by ttres on 22/06/2021.
//

#ifndef ML_LIBRARY_RBF_H
#define ML_LIBRARY_RBF_H

#include <vector>
#include "Centroid.h"
#include <cmath>

#define MAX_DIM_GAP_ALLOWED 0.001
#define MAX_GAP_ALLOWED 0.000001
Centroid** kmeans(double **X, int len_X,const int input_dim,const int k ,const int max_iters = 200);
Centroid** init_kmeans(const int cluster_count, double** dataset , const int dataset_size, const int input_dim);

double *predict_kmeans(Centroid **clusters, const int k, const float *X);
double *predict_rbfn(Centroid **clusters, const int k, const float *X);
void destroy_rbfn_prediction(const double* prediction);
MatrixXd train_rbfn_model(double **flattened_dataset_inputs,
                          int samples_count,
                          double **flattened_dataset_expected_outputs,
                          const int input_dim,
                          const int k,
                          const int max_iters=100);



#endif //ML_LIBRARY_RBF_H
