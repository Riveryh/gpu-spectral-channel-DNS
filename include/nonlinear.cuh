#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data.h"
#include "transform.cuh"

__host__ int getNonlinear(problem& pb);
__host__ int computeLambVector(problem& pb);
__host__ int rhsNonlinear(problem& pb);
__global__ void rhsNonlinearKernel(cudaPitchedPtrList plist,
	int mx, int my, int mz, REAL alpha, REAL beta);