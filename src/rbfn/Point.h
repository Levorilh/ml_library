//
// Created by Réda Maizate on 22/06/2021.
//

#ifndef ML_LIBRARY_POINT_H
#define ML_LIBRARY_POINT_H

#include "../headers/main.h"

class Point {
private:
    int label;
    float *coords;
    int coord_count;

public:
    Point();
    Point(int label, float *coords, int coord_count);
    double distance_to(Point* target);
};
#endif //ML_LIBRARY_POINT_H
