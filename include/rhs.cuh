#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data.h"

//__host__ int getRHS(problem& pb);

__host__ int get_rhs_v(problem& pb);
__host__ int get_rhs_omega(problem& pb);
void launch_subthread(problem& pb);
void terminate_subthread();
void synchronizeGPUsolver();


