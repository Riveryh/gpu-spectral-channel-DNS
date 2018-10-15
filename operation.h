#pragma once
#include "data.h"
#include <string>

namespace RPCF{
	//int read_parameter(Flow& fl, std::string& s);
	int write_3d_to_file(char* filename, real* pu, int pitch, int nx, int ny, int nz);
};

__host__ __device__
void get_ialpha_ibeta(int kx, int ky, int my,
	real alpha, real beta,
	real& ialpha, real& ibeta);

enum myCudaMemType {
	XYZ_3D,
	ZXY_3D,
	FREE_3D
};


__host__ void initMyCudaMalloc(dim3 dims);
__host__ cudaError_t myCudaMalloc(cudaPitchedPtr& Ptr, myCudaMemType type);
__host__ cudaError_t myCudaFree(cudaPitchedPtr& Ptr, myCudaMemType type);
__host__ void destroyMyCudaMalloc();
__host__ void* get_fft_buffer_ptr();