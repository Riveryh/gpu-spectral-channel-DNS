#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data.h"
#include <stdio.h>

template<class T>
class cuMat3d{
public:
	T* mat;
	cuMat3d<T>(int i) {
		size_t size = 5 * sizeof(T);
		cudaMalloc(&mat, size);
		mat[0] = (T)i;
	};
};



__host__ int initCUDA(problem& pb);
__host__ void setupCUDA(problem& pb);

__host__ int initFlow(problem& pb);

//__host__ int computeNonlinear(problem& pb);

__host__ __device__ void ddz(real* u, int N);
__host__ __device__ void ddz(complex* u, int N);


__device__ void ddz_sm(real* u, int N, int kz);
__device__ void ddz_sm(complex* u, int N, int kz);


