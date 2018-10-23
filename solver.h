#include "data.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int nextStep(problem& pb);

int initSolver(problem& pb, bool inversed = true);

int destroySolver(problem& pb);

__host__ cudaError_t m_multi_v_gpu(complex* _mat, complex* v, const int N, const size_t pitch, const int batch);