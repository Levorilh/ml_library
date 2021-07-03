//
// Created by Réda Maizate on 22/06/2021.
//

#ifndef ML_LIBRARY_POINT_H
#define ML_LIBRARY_POINT_H

#include "../main.h"
#include <vector>

class Point {
public:
    Point();
    Point(int label, float *coords, int coord_count);
    double distance_to(Point *target);
    double distance_to(float *data_line);
    void toString();
    int getLabel();
    int getCoord_count();
    void updateSTD(vector<float*> data);

    int label;
    float *coords;
    int coord_count;
    double deviation;

private:
};

#endif //ML_LIBRARY_POINT_H