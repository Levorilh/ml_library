#ifndef LIBRARY_MAIN_H

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../../Eigen/Dense"
#include "utils.h"

#include "linear/linear.h"
#include "mlp/MLP.h"
#include "rbfn/Point.h"


#define LIBRARY_MAIN_H
#endif //LIBRARY_MAIN_H
