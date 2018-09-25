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