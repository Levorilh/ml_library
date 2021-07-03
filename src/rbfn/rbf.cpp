#include "../headers/rbfn/rbf.h"

using namespace std;

float mean(vector<float*> group,int coord_to_mean) {
    float sum = 0;
    for(int i = 0; i < group.size();  i+=1){
        sum += group.at(i)[coord_to_mean];
    }
//    float sum = accumulate(group.begin(), group.end(), 0.0);
    float mean = sum / group.size();
    return mean;
}

Point** init_kmeans(const int cluster_count, float** dataset , const int dataset_size, const int input_dim){
    auto centroids = new Point *[cluster_count];
    for (int i = 0; i < cluster_count; i++) {
        centroids[i] = new Point(i, dataset[rand() % dataset_size], input_dim);
    }

    return centroids;
}

Point** kmeans(float **X, int len_X,const int input_dim,const int k,const int max_iters) {
    Point** clusters = init_kmeans(k , X , len_X , input_dim);

//    for (int i = 0; i < k; i++) {
//        (*clusters[i]).toString();
//    }

    bool converged = false;
    int current_iter = 0;
    double gap;
    while ((!converged) && (current_iter < max_iters)) {
        converged = true;
        vector<vector<float*>> clustered_data(k);

        for (int x = 0; x < len_X; x++) {
            vector<double> distances(k);
            int label_min_distance = 0;
            float* current_elem = X[x];
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
                float curr_dim = clusters[i]->coords[j];

                clusters[i]->coords[j] = mean(clustered_data[clusters[i]->label]  ,j );
                dim_gap = abs(curr_dim - clusters[i]->coords[j]);

                if(dim_gap >= MAX_DIM_GAP_ALLOWED){
                    converged = false;
                }
                gap += dim_gap;
            }
            clusters[i]->updateSTD(clustered_data[clusters[i]->label]);
        }
        if(gap < MAX_GAP_ALLOWED){
            converged = true;
        }
        current_iter += 1;
    }
    return clusters;
}

//float* predict_rbfn(Point** clusters , const int k , const float* X, const int X_size){
//
//    float* prediction = new float[k];
//    for(int i = 0 ; i < k ; i+=1){
//
//        float distance  = clusters[i]->distance_to(X);
//
//        prediction[i] = 1 / exp(-distance / ))
////                1 / np.exp(-distance / s ** 2)
//    }
//
//
//
//    return prediction;
//
//}

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