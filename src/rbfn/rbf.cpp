#include <cmath>
#include <algorithm>
#include "../headers/rbfn/Point.h"
#include <vector>

Point** init_kmeans(const int cluster_count, float** dataset , const int dataset_size, const int input_dim){
    auto centroids = new Point *[cluster_count];
    for (int i = 0; i < cluster_count; i++) {
        centroids[i] = new Point(rand() % cluster_count, dataset[rand() % dataset_size], input_dim);
    }

    return centroids;
}


Point** kmeans(float **X, int len_X, int input_dim,int k, int max_iters) {
    cout << len_X << "   " << k << endl;
    Point** clusters = init_kmeans(k , X , len_X , input_dim);

    for (int i = 0; i < k; i++) {
        (*clusters[i]).toString();
    }

    bool converged = false;
    int current_iter = 0;

    while ((!converged) && (current_iter < max_iters)) {
        auto clusterized_data = new std::vector<float*>[k];
        for (int x = 0; x < len_X; x++) {
            auto distances = new double[k];
            int label_min_distance = 0;
            float* current_elem = X[x];
            for(int cluster_pos = 0; cluster_pos < k ; cluster_pos +=1){

                distances[cluster_pos] = clusters[cluster_pos]->distance_to(current_elem);
                if( distances[cluster_pos] < distances[label_min_distance]){
                    label_min_distance = cluster_pos;
                }
            }
            clusterized_data[label_min_distance].push_back(current_elem);
            cout << label_min_distance << endl;
            delete[] distances;
        }
        delete[] clusterized_data;
        current_iter += 1;


    }
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

    return clusters;

}