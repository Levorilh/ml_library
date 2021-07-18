#include "../headers/rbfn/Centroid.h"

using namespace std;

Centroid::Centroid() {
    label = -1;
    coords = nullptr;
    coord_count = -1;
    deviation = 0;
}

Centroid::Centroid(int label, double *coords, int coord_count) {
    this->label = label;
    this->coords = (double*)malloc(sizeof(double)*coord_count);
    this->coords = coords;
    this->coord_count = coord_count;
    this->deviation = 0;
}

double Centroid::distance_to(Centroid *target) {
    return distance_to(target->coords);
}

double Centroid::distance_to(const double *data_line) {
    double sum = 0.;
    for (int i = 0; i < coord_count; i++) {
        sum += pow((coords[i] - data_line[i]), 2);
    }

//    cout << "distance from [" << coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << "] to ["
//         << data_line[0] << " " << data_line[1] << " " << data_line[2] << " " << data_line[3] << "] =  "
//         << sqrt(sum) << endl;
    return sqrt(sum);
}


void Centroid::toString() {
    cout << "label : " << label << endl;
    for (int i = 0; i < coord_count; ++i) {
        cout << "entree " << i << " : " << coords[i] << endl;
    }
}

int Centroid::getLabel() {
    return label;
}

int Centroid::getCoord_count(){
    return coord_count;
}

void Centroid::updateSTD(vector<double*> closests_points){

    int data_size = (int)(closests_points.size()*coord_count);
    double sum = 0;
    for ( int i = 0 ; i < closests_points.size() ; i +=1){
        for(int j = 0 ; j < coord_count; j++){
            sum += closests_points[i][j];

        }
    }
    double mean = sum / data_size;

    double variance = 0.;
    for ( int i = 0 ; i < closests_points.size() ; i +=1){
        for(int j = 0 ; j < coord_count; j++){
            variance += pow(closests_points[i][j] - mean, 2);
        }
    }
    variance /= data_size;

    deviation = sqrt(variance);
//    printf("deviation %d: %lf\n\n" ,label, deviation);

}