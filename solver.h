#include "data.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int nextStep(problem& pb);

int initSolver(problem& pb);

int destroySolver(problem& pb);