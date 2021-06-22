//
// Created by Réda Maizate on 22/06/2021.
//
#include <new>
#include <cmath>
#include <random>
#include "Point.h"
#include "../headers/utils.h"

Point *kmeans(float **X, int len_X, int input_dim, int k, int max_iters) {

    auto centroids = new Point[k];
    for (int i = 0; i < k; i++) {
        centroids[i] = Point(rand() % k, X[rand() % len_X], input_dim);
    }

//    for (int i = 0; i < k; i++) {
//        centroids[i] = X[rand() % len_X];
//    }

//    bool converged = false;
//    current_iter = 0;
//
//    while ((!converged) && (current_iter < max_iters)) {
//        // cluster_list = [[] for i in range(len(centroids))]
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

    return centroids;

}