#pragma once

//Macro for checking cuda errors following a cuda launch or api call
#ifndef CUDA_CHECK_ERROR(R)
#define CUDA_CHECK_ERROR(R) {											                       \
 cudaError_t e=cudaGetLastError();                                                         \
 if(e!=cudaSuccess) {                                                                      \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   return R;																			   \
 }																						   \
}
#endif

struct util
{
	static void clear_screen();
};
