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

Centroid** init_kmeans(const int cluster_count, double** dataset , const int dataset_size, const int input_dim){
    auto centroids = new Centroid *[cluster_count];
    for (int i = 0; i < cluster_count; i++) {
        centroids[i] = new Centroid(i, dataset[rand() % dataset_size], input_dim);
    }

    return centroids;
}

Centroid** kmeans(double **X, int len_X,const int input_dim,const int k,const int max_iters) {
    Centroid** clusters = init_kmeans(k , X , len_X , input_dim);

//    for (int i = 0; i < k; i++) {
//        (*clusters[i]).toString();
//    }

    bool converged = false;
    int current_iter = 0;
    double gap;
    while ((!converged) && (current_iter < max_iters)) {
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

        gap = 0;
        for(int i = 0; i < k ; i +=1){
            for(int j = 0; j < clusters[i]->coord_count ; j+=1){
                double dim_gap;
                double curr_dim = clusters[i]->coords[j];

                clusters[i]->coords[j] = mean(clustered_data[clusters[i]->label]  ,j );
                dim_gap = abs(curr_dim - clusters[i]->coords[j]);


                gap += dim_gap;
            }
            clusters[i]->updateSTD(clustered_data[clusters[i]->label]);
        }
        if(gap == 0){
            converged = true;
        }
        current_iter += 1;
    }
    return clusters;
}

double *predict_rbfn(Centroid **clusters, const int k, const double *X) {

    auto* prediction = new double[k];
    int s = 2;
    for(int i = 0 ; i < k ; i+=1){

        double distance  = clusters[i]->distance_to(X);

        prediction[i] = 1. / exp(-distance / pow(clusters[i]->deviation,2));
    }
    return prediction;
}

void destroy_rbfn_prediction(const double* prediction){
    delete prediction;
}
/*
    pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

    print('K-MEANS: ', int(pattern))

    converged = (pattern == 0)

    current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]

*/


//        vector cluster_list = [[] for i in range(len(centroids))]
//        for (int x = 0; x < len_X; x++) {
//            // distances_list = []
//            for (int c = 0; c < k; c++) {
//                distances_list[c] = get_distance(c, len_X, x);
//            }
//            // cluster_list[int(np.argmin(distances_list))].append(x)
//        }
//
//        // cluster_list = list((filter(None, cluster_list)))
//        float *new_centroids[k];
//    }