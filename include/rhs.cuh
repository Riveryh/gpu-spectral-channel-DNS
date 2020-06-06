#include "data.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//__host__ int getRHS(problem& pb);

__host__ int get_rhs_v(problem& pb);
__host__ int get_rhs_omega(problem& pb);
void init_pthread(problem& pb);


