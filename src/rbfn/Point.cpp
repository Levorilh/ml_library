#include "Point.h"

//Point::Point(){
//    label= -1;
//    coords = nullptr;
//    coord_count = -1;
//}

Point::Point(int label, float *coords , int coord_count){
        this->label = label;
        this->coords = coords;
        this->coord_count = coord_count;
}

double Point::distance_to(Point* target) {
    double sum = 0.;
    for (int i = 0; i < coord_count; i++) {
        sum += pow((coords[i] - target->coords[i]), 2);
    }
    return sqrt(sum);
}
