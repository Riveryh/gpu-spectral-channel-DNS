#pragma once

#include <string>

#include "data.h"
#include "RPCFKernels.cuh"
#include "operation.h"
#include "transform.cuh"

//check the return value of cuda calls.
void cuCheck(cudaError_t ret, std::string s = "");


//#define CURPCF_CUDA_PROFILING

