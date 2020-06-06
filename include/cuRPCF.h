#pragma once
#include "data.h"
#include "RPCFKernels.cuh"
#include "operation.h"
#include "transform.cuh"
#include <string>

//check the return value of cuda calls.
void cuCheck(cudaError_t ret, std::string s = "");


//#define CURPCF_CUDA_PROFILING

