#ifndef ML_LIBRARY_MLP_H
#define ML_LIBRARY_MLP_H


class MLP{
public:
    int *d;
    float** X;
    float** deltas;
    float*** W;
    int d_length;
};

typedef class MLP MLP;

#endif //ML_LIBRARY_MLP_H
