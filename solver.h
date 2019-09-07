#include "data.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int nextStep(problem& pb);

int initSolver(problem& pb, bool inversed = true);

int destroySolver(problem& pb);

int solveEq(complex* inv_coef, complex* rhs, int N,
	size_t pitch, int nx, int ny);
int solveEqGPU(complex* inv_coef, complex* rhs, int N,
	size_t pitch, int mx, int my, int num_equation);
void save_0_v_omega_y(problem& pb);
void synchronizeGPUsolver();

__host__ cudaError_t m_multi_v_gpu(complex* _mat, complex* v, const int N, const size_t pitch, const int batch);