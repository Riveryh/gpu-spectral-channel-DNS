#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "data.h"

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



__host__ int allocDeviceMem(problem& pb);
__host__ void getDeviceInfo(problem& pb);
__host__ int initFlow(problem& pb);

//__host__ int computeNonlinear(problem& pb);

__host__ __device__ inline void ddz(REAL* u, int N) {
	REAL buffer[MAX_NZ * 4];
	REAL dmat;
	for (int i = 0; i < N; i++) {
		buffer[i] = 0;
		for (int j = i + 1; j < N; j = j + 2) {
			dmat = 2 * (j);
			buffer[i] = buffer[i] + dmat * u[j];
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}
__host__ __device__ inline void ddz(cuRPCF::complex* u, int N) {
	cuRPCF::complex buffer[MAX_NZ];
	REAL dmat;
	cuRPCF::complex buffer_u[MAX_NZ];
	for (int i = 0; i < N; i++) {
		buffer_u[i] = u[i];
	}
	for (int i = 0; i < N; i++) {
		buffer[i] = cuRPCF::complex(0.0, 0.0);
		for (int j = i + 1; j < N; j = j + 2) {
			dmat = 2 * REAL(j);
			buffer[i] = buffer[i] + buffer_u[j] * dmat;
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}

#ifdef __NVCC__

__device__ __forceinline__ void ddz_sm(REAL* u, int N, int kz) {
	REAL buffer;
	REAL dmat;

	//wait all threads to load data before computing
	__syncthreads();

	buffer = 0.0;
	for (int j = kz + 1; j < N; j = j + 2) {
		dmat = 2 * REAL(j);
		buffer = buffer + u[j] * dmat;
	}
	//wait all threads to finish computation before overwriting array.
	__syncthreads();
	if (kz == 0) {
		u[0] = buffer * 0.5;
	}
	else
	{
		u[kz] = buffer;
	}
}
__device__ __forceinline__ void ddz_sm(cuRPCF::complex* u, int N, int kz) {
	cuRPCF::complex buffer;
	REAL dmat;

	//wait all threads to load data before computing
	__syncthreads();

	buffer = cuRPCF::complex(0.0, 0.0);
	for (int j = kz + 1; j < N; j = j + 2) {
		dmat = 2 * REAL(j);
		buffer = buffer + u[j] * dmat;
	}
	//wait all threads to finish computation before overwriting array.
	__syncthreads();
	if (kz == 0) {
		u[0] = buffer * 0.5;
	}
	else
	{
		u[kz] = buffer;
	}
}

#endif
