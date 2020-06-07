#pragma once
#include <cufft.h>

#include "transpose.cuh"

#define NUM_GPU 1
extern cufftHandle planXYr2c_M[NUM_GPU], planXYc2r_M[NUM_GPU], planZ_pad_M[NUM_GPU];
extern int dev_id[NUM_GPU];

//int transform_one_mGPU(DIRECTION dir, cudaPitchedPtr& Ptr, cudaPitchedPtr& tPtr, dim3 dims, int dev_id);