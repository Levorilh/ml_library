#include "../headers/rbfn/rbf.h"

using namespace std;


double mean(vector<double*> group,int coord_to_mean) {
    double sum = 0;
    for(int i = 0; i < group.size();  i+=1){
        sum += group.at(i)[coord_to_mean];
    }
//    float sum = accumulate(group.begin(), group.end(), 0.0);
    double mean = sum / (double)(group.size());
    return mean;
}

Centroid** init_kmeans(const int cluster_count, double** dataset , const int dataset_size, const int input_dim, bool naif){
    Centroid ** centroids = (Centroid **)malloc(sizeof(Centroid *) * cluster_count);//new Centroid *[cluster_count];
    for (int i = 0; i < cluster_count; i++) {
        centroids[i] = (Centroid *)malloc(sizeof(Centroid));
        double * dataset_row_copy = (double*)malloc(sizeof(double)*input_dim);
        int random_index = rand() % dataset_size;
        for(int j = 0; j < input_dim; j++){
            if (naif)
                random_index = i;
            dataset_row_copy[j] = dataset[random_index][j];
        }
        centroids[i] = new Centroid(i, dataset_row_copy/*dataset[rand() % dataset_size]*/, input_dim);
    }

    return centroids;
}

Centroid** train_kmeans(double **X, int len_X,const int input_dim,const int k, bool naif ,const int max_iters) {
    Centroid** clusters = init_kmeans(k , X , len_X , input_dim, naif);

    bool converged = false;
    int current_iter = 0;
    double gap;
    while ((!converged) && (current_iter < max_iters)) {
        printf("iter : %d/%d\n",  current_iter+1, max_iters);
        vector<vector<double*>> clustered_data(k);

        for (int x = 0; x < len_X; x++) {
            vector<double> distances(k);
            int label_min_distance = 0;
            double* current_elem = X[x];
            for(int cluster_pos = 0; cluster_pos < k ; cluster_pos +=1){

                distances[cluster_pos] = clusters[cluster_pos]->distance_to(current_elem);
                if( distances[cluster_pos] < distances[label_min_distance]){
                    label_min_distance = clusters[cluster_pos]->getLabel();
                }
            }

            clustered_data[label_min_distance].push_back(current_elem);
//            cout << label_min_distance << endl;
        }
        for(int i = 0; i < k ; i++){
            cout << "points pour cluster " << i <<" : " << clustered_data[0].size() << endl;
        }
        gap = 0.;
        for(int i = 0; i < k ; i +=1){
            for(int j = 0; j < clusters[i]->coord_count ; j+=1){
                double dim_gap;
                double curr_dim = clusters[i]->coords[j];

                clusters[i]->coords[j] = mean(clustered_data[clusters[i]->label]  ,j );
                dim_gap = abs(curr_dim - clusters[i]->coords[j]);


                gap += dim_gap;
                printf("cluster %d coord %d / somme des ecarts : %lf\n" ,i  ,j ,dim_gap);
            }
            clusters[i]->updateSTD(clustered_data[clusters[i]->label]);
        }
        if(gap < MAX_GAP_ALLOWED ){
            printf("sortie prematuree\n");
            converged = true;
        }
        current_iter += 1;
    }
    return clusters;
}

double *predict_kmeans(Centroid **clusters, const int k, const double *X) {

    double* prediction = (double*)malloc(sizeof(double) * k);

    for(int i = 0 ; i < k ; i+=1){

        double distance  = clusters[i]->distance_to(X);

        prediction[i] = 1. / exp(-distance / 10);//pow(clusters[i]->deviation,2));
    }
    return prediction;
}

DLLEXPORT void destroy_rbfn_prediction(double* prediction){
    free(prediction);
}

DLLEXPORT RBF * create_rbfn_model(int input_dim, int num_classes, int k){
    RBF * rbf = (RBF*)malloc(sizeof(RBF));

    rbf->num_classes = num_classes;
    rbf->input_dim = input_dim;
    rbf->k = k;
    rbf->samples_count = 0;

    rbf->clusters = (Centroid **)malloc(sizeof(Centroid *) * k);
    for(int i = 0; i < k; i++){
        rbf->clusters[i] = (Centroid *)malloc(sizeof(Centroid));
    }

    rbf->W = (double**)malloc(sizeof(double*) * k);
    for(int i = 0; i < k; i++){
        rbf->W[i] = (double*)malloc(sizeof(double) * num_classes);
        for(int j = 0; j < num_classes; j++){
            rbf->W[i][j] = 0;
        }
    }

    return rbf;
}

DLLEXPORT void destroy_rbfn_model(RBF* model){
    for(int i = 0; i < model->input_dim; i++){
        free(model->W[i]);
    }
    free(model->W);
    for(int i = 0; i < model->k; i++){
        free(model->clusters[i]);
    }
    free(model->clusters);
    free(model);
}

DLLEXPORT void train_rbfn_model(RBF * model,
                      double *flattened_dataset_inputs,
                      int samples_count,
                      double *flattened_dataset_expected_outputs,
                      bool naif,
                      const int max_iters) {
    model->samples_count = samples_count;

    double ** dataset_inputs = (double**)malloc(sizeof(double*) * samples_count);
    double ** dataset_expected_outputs = (double**)malloc(sizeof(double*) * samples_count);
    for(int i = 0; i < samples_count; i++){
        dataset_inputs[i] = (double*)malloc(sizeof(double) * model->input_dim);
        for(int j = 0; j < model->input_dim; j++){
            dataset_inputs[i][j] = flattened_dataset_inputs[j + i * model->input_dim];
            //cout << "dataset_inputs ["<<i<<","<<j<<"] :" << dataset_inputs[i][j] << " | " << j + i * model->input_dim << endl;
        }
        dataset_expected_outputs[i] = (double*)malloc(sizeof(double) * model->num_classes);
        for(int j = 0; j < model->num_classes; j++){
            dataset_expected_outputs[i][j] = flattened_dataset_expected_outputs[j + i * model->num_classes];
            //cout << "dataset_expected_outputs ["<<i<<","<<j<<"] :" << dataset_expected_outputs[i][j] << endl;
        }
    }

    model->clusters = train_kmeans(dataset_inputs, samples_count, model->input_dim, model->k, naif, max_iters);

    MatrixXd X(samples_count, model->k);
    for (int i = 0; i < samples_count; i++) {

        auto* RBF_X = predict_kmeans(model->clusters, model->k, dataset_inputs[i]);
        for(int j = 0 ; j < model->k; j += 1){
            X(i,j) = RBF_X[j];
            //cout <<"X[" << i << ":" << j << "]=" << X(i,j) << endl;
        }
        destroy_rbfn_prediction(RBF_X);
    }
    //cout << "Sondage cluster train : " << model->clusters[0]->coords[0] << endl;
    MatrixXd Y(samples_count, model->num_classes);
    for (int i = 0; i < samples_count; i++) {
        for (int j = 0; j < model->num_classes; j++) { // pourquoi k ? devrais ??tre num_clases ?
            Y(i, j) = dataset_expected_outputs[i][j];
            //cout << "Y[" << i << ":" << j << "]=" << Y(i, j) << endl;
        }
    }
    //(input_dim / sample_count) * (sample_cout / input_dim) * (input_dim / sample_count) * (sample_count / num_classes) = (input_dim / num_classes)
    MatrixXd W = (((X.transpose() * X).inverse()) * X.transpose()) * Y;
    //cout << "W rows = " << W.rows() << "; clos = " << W.cols() << ";" << endl;
    for(int i=0; i < W.rows(); i++){
        for(int j=0; j < W.cols(); j++){
            model->W[i][j] = W(i, j);
            //cout << "W[" << i << ":" << j << "]=" << W(i, j) << endl;
        }
    }

    for(int i = 0; i < samples_count; i++){
        free(dataset_inputs[i]);
        free(dataset_expected_outputs[i]);
    }
    free(dataset_inputs);
    free(dataset_expected_outputs);
}

DLLEXPORT double *predict_rbfn(RBF *model, double *flattened_dataset_inputs) {
    auto* prediction = (double*)malloc(sizeof(double) * model->num_classes);
    MatrixXd X(1, model->k);
    //cout << "Sondage cluster predict : " << model->clusters[0]->coords[0] << endl;
    auto* RBF_X = predict_kmeans(model->clusters, model->k, flattened_dataset_inputs);
    for(int j = 0 ; j < model->k; j += 1){
        X(0,j) = RBF_X[j];
        //cout << "RBF X ["<<j<<"] : "<< RBF_X[j]<<endl;
    }
    free(RBF_X);

    MatrixXd W(model->k, model->num_classes);
    for(int i =0; i < model->k; i++){
        for(int j =0; j < model->num_classes; j++){
            W(i,j)=model->W[i][j];
        }
    }

    MatrixXd rslt = X * W;
    for(int i =0; i < rslt.cols(); i++){
        prediction[i] = rslt(0,i);
        //cout << "predict["<<i<<"] = "<<prediction[i]<<endl;
    }

    return prediction;
}

DLLEXPORT void save_rbf_model(RBF * model, const char* path){
    FILE* fp = fopen( path, "w" );
    fprintf(fp, "%d;%d;%d;%d\n", model->input_dim, model->num_classes, model->k, model->samples_count);
    for(int i = 0; i < model->k; i++){
        for(int j = 0; j < model->num_classes; j++){
            fprintf(fp, "%f\n",model->W[i][j]);
        }
    }
    for(int i = 0; i < model->k; i++){
        fprintf(fp, "%d;%d\n", model->clusters[i]->label,  model->clusters[i]->coord_count);
        for(int j = 0; j < model->clusters[i]->coord_count; j++){
            fprintf(fp, "%f\n",model->clusters[i]->coords[j]);
        }
    }
    fclose(fp);
}

DLLEXPORT RBF * load_rbf_model(const char* path){
    const int maxLength = 250;
    FILE *fp;
    fp = fopen(path , "r");
    if(!fp){
        return nullptr;
    }
    char* model_to_string = (char*)malloc(sizeof(char) * maxLength);
    int * len = (int*)malloc(sizeof(int));

    fgets(model_to_string, maxLength,fp);
    char ** s = split(model_to_string, len);
    int input_dim=strtol(s[0], nullptr, 10);
    int num_classes=strtol(s[1], nullptr, 10);
    int k=strtol(s[2], nullptr, 10);

    RBF * result = create_rbfn_model(input_dim, num_classes, k);

    result->samples_count=strtol(s[3], nullptr, 10);

    free(s[0]);
    free(s[1]);
    free(s[2]);
    free(s[3]);
    free(s);

    for(int i = 0; i < result->k; i++){
        for(int j = 0; j < result->num_classes; j++){
            fgets(model_to_string, maxLength,fp);
            result->W[i][j] = strtod(model_to_string, nullptr);
        }
    }

    for(int i = 0; i < result->k; i++){
        fgets(model_to_string, maxLength,fp);
        char ** s = split(model_to_string, len);
        result->clusters[i]->label = strtol(s[0], nullptr, 10);
        result->clusters[i]->coord_count = strtol(s[1], nullptr, 10);
        free(s[0]);
        free(s[1]);
        free(s);
        result->clusters[i]->coords = (double*)malloc(sizeof(double) * result->clusters[i]->coord_count);
        for(int j = 0; j < result->clusters[i]->coord_count; j++){
            fgets(model_to_string, maxLength,fp);
            result->clusters[i]->coords[j] = strtod(model_to_string, nullptr);
        }
    }

    fclose(fp);
    free(model_to_string);
    free(len);
    return result;

}