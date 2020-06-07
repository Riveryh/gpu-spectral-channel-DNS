#pragma once
#include <string>

#include "data.h"

namespace RPCF{
	//int read_parameter(Flow& fl, std::string& s);
	int write_3d_to_file(char* filename, REAL* pu, int pitch, int nx, int ny, int nz);
};

__host__ __device__
void get_ialpha_ibeta(int kx, int ky, int my,
	REAL alpha, REAL beta,
	REAL& ialpha, REAL& ibeta);

enum myCudaMemType {
	XYZ_3D,
	ZXY_3D,
	FREE_3D
};


__host__ void initMyCudaMalloc(dim3 dims);
__host__ cudaError_t myCudaMalloc(cudaPitchedPtr& Ptr, myCudaMemType type, int dev_id = 0);
__host__ cudaError_t myCudaFree(cudaPitchedPtr& Ptr, myCudaMemType type, int dev_id = 0);
__host__ void destroyMyCudaMalloc(int dev_id = 0);
__host__ void* get_fft_buffer_ptr(int dev_id = 0);