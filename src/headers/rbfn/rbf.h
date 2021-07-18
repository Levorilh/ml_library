//
// Created by ttres on 22/06/2021.
//

#ifndef ML_LIBRARY_RBF_H
#define ML_LIBRARY_RBF_H

#include <vector>
#include "Centroid.h"
#include "../main.h"
#include <cmath>

#define MAX_DIM_GAP_ALLOWED 0.001
#define MAX_GAP_ALLOWED 0.000001

DLLEXPORT class RBF{
public :
    int input_dim;
    int num_classes;
    int k;
    int samples_count;
    double ** W;
    Centroid ** clusters;

};

Centroid** train_kmeans(double **X, int len_X,const int input_dim,const int k ,const int max_iters = 200);
Centroid** init_kmeans(const int cluster_count, double** dataset , const int dataset_size, const int input_dim);

double *predict_kmeans(Centroid **clusters, const int k, const float *X);
DLLEXPORT RBF * create_rbfn_model(int input_dim, int num_classes, int k);
DLLEXPORT void destroy_rbfn_model(RBF* model);
DLLEXPORT double *predict_rbfn(RBF *model, double *flattened_dataset_inputs);
DLLEXPORT void destroy_rbfn_prediction(const double* prediction);
DLLEXPORT void train_rbfn_model(RBF* model,
                      double *flattened_dataset_inputs,
                      int samples_count,
                      double *flattened_dataset_expected_outputs,
                      const int max_iters=100);



#endif //ML_LIBRARY_RBF_H
