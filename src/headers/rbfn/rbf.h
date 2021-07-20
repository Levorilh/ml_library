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

class RBF{
public :
    int input_dim;
    int num_classes;
    int k;
    int samples_count;
    double ** W;
    Centroid ** clusters;

};

Centroid** train_kmeans(double **X, int len_X,const int input_dim,const int k , bool naif, const int max_iters = 200);
Centroid** init_kmeans(const int cluster_count, double** dataset , const int dataset_size, const int input_dim , bool naif);

double *predict_kmeans(Centroid **clusters, const int k, const float *X);
DLLEXPORT RBF * create_rbfn_model(int input_dim, int num_classes, int k);
DLLEXPORT void destroy_rbfn_model(RBF* model);
DLLEXPORT double *predict_rbfn(RBF *model, double *flattened_dataset_inputs);
DLLEXPORT void destroy_rbfn_prediction(double* prediction);
DLLEXPORT void train_rbfn_model(RBF* model,
                      double *flattened_dataset_inputs,
                      int samples_count,
                      double *flattened_dataset_expected_outputs,
                      bool naif=false,
                      const int max_iters=100);
DLLEXPORT void save_rbf_model(RBF * model, const char* path);
DLLEXPORT RBF * load_rbf_model(const char* path);

#endif //ML_LIBRARY_RBF_H
