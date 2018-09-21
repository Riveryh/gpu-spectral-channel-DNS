#pragma once
#include "data.h"
#include <string>

namespace RPCF{
	//int read_parameter(Flow& fl, std::string& s);
	int write_3d_to_file(char* filename, real* pu, int pitch, int nx, int ny, int nz);
};
