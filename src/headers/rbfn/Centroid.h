//
// Created by Réda Maizate on 22/06/2021.
//

#ifndef ML_LIBRARY_CENTROID_H
#define ML_LIBRARY_CENTROID_H

#include "../main.h"
#include <vector>

DLLEXPORT class Centroid {
public:
    Centroid();
    Centroid(int label, double *coords, int coord_count);
    double distance_to(Centroid *target);
    double distance_to(const double *data_line);
    void toString();
    int getLabel();
    int getCoord_count();
    void updateSTD(vector<double*> data);

    int label;
    double *coords;
    int coord_count;
    double deviation;

private:
};

#endif //ML_LIBRARY_CENTROID_H
