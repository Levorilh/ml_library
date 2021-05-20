#ifndef LIBRARY_MAIN_H

#if defined( _WIN32) || defined( __WIN32__) || defined( WIN32 ) || defined( __NT__ )
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif


#include "linear/create.h"
//#include "mlp/MLP_model.h"



#define LIBRARY_MAIN_H
#endif //LIBRARY_MAIN_H
