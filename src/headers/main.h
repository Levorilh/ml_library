#ifndef LIBRARY_MAIN_H

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../../Eigen/Dense"
#include "Linear_model.h"
#include "MLP_model.h"



#define LIBRARY_MAIN_H
#endif //LIBRARY_MAIN_H
