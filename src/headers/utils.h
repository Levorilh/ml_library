#ifndef ML_LIBRARY_UTILS_H
#define ML_LIBRARY_UTILS_H
    #include "main.h"
    DLLEXPORT void init_random();
    char ** split(char* str, int* array_len, char delimiter = ';');
#endif //ML_LIBRARY_UTILS_H
