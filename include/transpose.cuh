#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data.h"

enum DIRECTION {
	FORWARD, BACKWARD
};

__host__ int transpose(DIRECTION dir, cudaPitchedPtr Ptr,
	cudaPitchedPtr tPtr, int* dim, int* tDim);

__host__ int cuda_transpose(DIRECTION dir, cudaPitchedPtr& input,
	cudaPitchedPtr& tPtr, int* dim, int* tDim);