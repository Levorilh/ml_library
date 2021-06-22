#include "../headers/rbfn/Point.h"

Point::Point(){
    label= -1;
    coords = nullptr;
    coord_count = -1;
}

Point::Point(int label, float *coords , int coord_count){
        this->label = label;
        this->coords = coords;
        this->coord_count = coord_count;
}

double Point::distance_to(Point* target) {
    return distance_to(target->coords);
}

double Point::distance_to(float *data_line) {
    double sum = 0.;
    for (int i = 0; i < coord_count; i++) {
        sum += pow((coords[i] - data_line[i]), 2);
    }

    cout << "distance from [" <<  coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << "] to [" << data_line[0] << " " << data_line[1] << " " << data_line[2] << " " << data_line[3] << "] =  " << sqrt(sum) << endl;
    return sqrt(sum);
}


void Point::toString(){
    cout << "label : " << label << endl;
    for (int i = 0; i < coord_count; ++i) {
        cout << "entree " << i << " : " << coords[i] << endl;
    }
}