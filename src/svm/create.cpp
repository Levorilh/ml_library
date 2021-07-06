#include "../headers/svm/create.h"

void create_svm_model(float* flattened_dataset_inputs, int input_dim, int samples_count, float* expected_output){
    //cout << "input_dim : " << input_dim << " | samples_count : " << samples_count << endl;
    MatrixXd X(samples_count, input_dim);
    for (int i = 0; i < samples_count; i++) {
        for (int j = 0; j < input_dim ; j++) {
            X(i, j) = flattened_dataset_inputs[i * input_dim + j];
        }
    }

    MatrixXd Y(samples_count, 1);
    for (int i = 0; i < samples_count; i++) {
        Y(i, 0) = expected_output[i];
    }

    MatrixXd BigMatrix(samples_count, samples_count);
    for (int i = 0; i < samples_count; i++) {
        MatrixXd Xi(input_dim,1);
        for (int j = 0; j < input_dim; j++){
            Xi(j, 0) = X(i, j);
        }
        for (int j = 0; j < samples_count ; j++) {
            MatrixXd Xj(input_dim, 1);
            for (int y = 0; y < input_dim; y++){
                Xj(y, 0) = X(j, y);
            }
            float tmp = 0;
            for(int t = 0; t < input_dim; t++){
                tmp += Xj(t,0) * Xi(t,0);
            }
            BigMatrix(i, j) = Y(i,0) * Y(j,0) * tmp;
            //cout << BigMatrix(i, j) << "|";
        }
        //cout << endl;
    }

    MatrixXd alphas(samples_count,1);
    for(int i = 0; i< samples_count; i++){
        float alpha = Y(i,0);
        alphas(i, 0) = alpha;
    }

    MatrixXd negatifs(1,samples_count);
    for(int i = 0; i< samples_count; i++){
        negatifs(0, i) = -1;
    }

    //TODO pas encore la fonction pour minimiser alpha
    float min = (1/2 * alphas.transpose() * BigMatrix * alphas + negatifs * alphas)(0,0);
    float Yt = (Y.transpose() * alphas)(0,0);
    cout << "min : " << min << endl;
    cout << "Yt.aplha : " << Yt << endl;
    for(int i = 0; i< samples_count; i++){
        cout << "aplha : " << alphas(i,0) << endl;
    }

    MatrixXd TmpW(1,input_dim);
    for(int i =1; i < samples_count; i++){
        MatrixXd Xi(1,input_dim);
        for(int y=0; y<input_dim; y++){
            Xi(0,y)=X(i,y);
        }
        TmpW += alphas(i,0) * Y(i,0) * Xi;
    }

    //TODO svPos reste a calculer
    int svPos = 0;

    float sumWX = 0;
    for(int i =0; i< input_dim; i++){
        sumWX += TmpW(0,i) * X(svPos, i) ;
    }

    float W0 = 1/Y(svPos,0) - sumWX;

    MatrixXd W(1,input_dim + 1);
    W(0,0) = W0;
    for(int i = 0; i < samples_count; i++){
        W(0, i+1) = TmpW(0, i);
    }
}