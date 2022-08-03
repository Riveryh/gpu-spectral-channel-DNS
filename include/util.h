#pragma once

#include <string>
#include <stdio.h>

#include "data.h"
#include "RPCFKernels.cuh"
#include "operation.h"
#include "transform.cuh"

//check the return value of cuda calls.
#ifndef NDEBUG
	#define CUDA_CHECK(res) __cuda_check(res, __FILE__, __LINE__)
#else
	#define CUDA_CHECK(res) (res)
#endif

inline void __cuda_check(cudaError_t res, const char* filename, int line) {
	if (res != cudaSuccess) {
		printf("CUDA error : %s at %s:%d\n", cudaGetErrorString(res), filename, line);
	}
}

//#define CURPCF_CUDA_PROFILING
//#define SHOW_TRANSFORM_TIME
