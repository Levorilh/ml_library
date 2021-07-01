//
// Created by N on 01/07/2021.
//

#include "../headers/tests/svm_model_test.h"

void test_svm_model(){
    int input_dim = 2;
    int samples_count = 5;

    float * flatten_X = (float*)malloc(sizeof(float) * input_dim * samples_count);
    flatten_X[0]=1.0;
    flatten_X[1]=1.0;
    flatten_X[2]=2.0;
    flatten_X[3]=1.0;
    flatten_X[4]=2.0;
    flatten_X[5]=2.0;
    flatten_X[6]=4.0;
    flatten_X[7]=1.0;
    flatten_X[8]=4.0;
    flatten_X[9]=4.0;

    float * flatten_Y = (float*)malloc(sizeof(float) * samples_count);
    flatten_Y[0]=1.0;
    flatten_Y[1]=1.0;
    flatten_Y[2]=-1.0;
    flatten_Y[3]=-1.0;
    flatten_Y[4]=-1.0;

    create_svm_model(flatten_X, input_dim, samples_count, flatten_Y);

    free(flatten_X);
    free(flatten_Y);
}
