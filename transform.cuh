#pragma once 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data.h"
#include "cufft.h"
#include "RPCFKernels.cuh"

extern cufftHandle planXYr2c, planXYc2r, planZ;

__global__ void investigate(cudaPitchedPtr p);

enum DIRECTION {
	FORWARD,BACKWARD
};
__host__ int transform_3d_one(DIRECTION dir, cudaPitchedPtr& input,
	cudaPitchedPtr& output, int* indim, int* outdim);

__host__ int initFFT(problem& pb);

__host__ int transform(DIRECTION dir,problem& pb);

__global__ void setZeros(cudaPitchedPtr p, int mx, int my, int mz);

__host__ int transpose(DIRECTION dir, cudaPitchedPtr input,
	cudaPitchedPtr output, int* indim, int* outdim);

__global__ void normalize(cudaPitchedPtr p, int mx, int my, int mz, real factor);

__host__ void cheby_p2s(cudaPitchedPtr tPtr, int mx, int my, int mz);
__host__ void cheby_s2p(cudaPitchedPtr tPtr, int mx, int my, int mz);

#define CACHE_SIZE 512
#define _MIN(x,y) (((x)<(y))?(x):(y))

#ifdef REAL_FLOAT
#define CUFFTEXEC_R2C cufftExecR2C
#define CUFFTEXEC_C2R cufftExecC2R
#define CUFFTEXEC_C2C cufftExecC2C
#define CUFFTREAL cufftReal
#define CUFFTCOMPLEX cufftComplex
#define myCUFFT_R2C CUFFT_R2C
#define myCUFFT_C2C CUFFT_C2C
#define myCUFFT_C2R CUFFT_C2R
#endif

#ifdef REAL_DOUBLE
#define CUFFTEXEC_R2C cufftExecD2Z
#define CUFFTEXEC_C2R cufftExecZ2D
#define CUFFTEXEC_C2C cufftExecZ2Z
#define CUFFTREAL cufftDoubleReal
#define CUFFTCOMPLEX cufftDoubleComplex
#define myCUFFT_R2C CUFFT_D2Z
#define myCUFFT_C2C CUFFT_Z2Z
#define myCUFFT_C2R CUFFT_Z2D
#endif

#define cuFFTcheck(x,s) if(x!=CUFFT_SUCCESS){fprintf(stderr,s);}
