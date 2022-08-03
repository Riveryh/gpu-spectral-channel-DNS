#include "cuda_runtime.h"
#include <cassert>

#include "../include/util.h"


__device__ __forceinline__ void _get_linear_v_device(
		cuRPCF::complex * rhs_v,
		cuRPCF::complex * nonlinear_v, 
		cuRPCF::complex * nonlinear_v_p,
		cuRPCF::complex * rhs_v_p,
		int N, REAL * U, REAL * ddU,
		REAL * T0, REAL * T2, REAL * T4,
		REAL Re, REAL dt, REAL kmn, REAL alpha)
{
	cuRPCF::complex* rhs_temp = (cuRPCF::complex*)malloc((N + 1) * sizeof(cuRPCF::complex));
	assert(rhs_temp != nullptr);
	for (int i = 4; i <= N; i++) {
		rhs_temp[i] = cuRPCF::complex(0.0, 0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_2_0 = (N + 1) * (i - 2) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_v[j] * cuRPCF::complex(
				T4[inc_2_0] * dt * 0.5 / Re + (1 - kmn * dt / Re) * T2[inc_2_0]
				- kmn * (1 - kmn * dt * 0.5 / Re) * T0[inc_2_0]
				//T2[inc_2_0] - kmn*T0[inc_2_0]
				,
				-alpha * dt * 0.5 * U[i - 2] * T2[inc_2_0]
				+ (kmn * alpha * dt * 0.5 * U[i - 2] + alpha * dt * 0.5 * ddU[i - 2]) * T0[inc_2_0]
			);
		}
	}

	//save new rhs data and add nonlinear part to it.
	for (int i = 0; i <= N; i++) {
		rhs_v_p[i] = rhs_v[i];	// save previous step v_hat data
		rhs_v[i] = rhs_temp[i];// +(nonlinear_v[i - 2] * 1.5 - nonlinear_v_p[i - 2] * 0.5)*dt;
	}

	//boundary conditions
	rhs_v[0] = cuRPCF::complex(0.0, 0.0);
	rhs_v[1] = cuRPCF::complex(0.0, 0.0);
	rhs_v[2] = cuRPCF::complex(0.0, 0.0);
	rhs_v[3] = cuRPCF::complex(0.0, 0.0);

	//remove pointer, no need on GPU.
	free(rhs_temp);
}

__global__ void get_linear_v_kernel(
	const int my,
	const int hnx,
	const int ny,
	const int nz,
	size_t tPitch,
	cuRPCF::complex* _rhs_v,
	cuRPCF::complex* _nonlinear_v,
	cuRPCF::complex* _nonlinear_v_p,
	cuRPCF::complex* _rhs_v_p,
	REAL aphi,
	REAL beta,
	REAL* _U0,
	REAL* _ddU0,
	REAL* T0, REAL* T2, REAL*T4,
	REAL Re, REAL dt
) 
{
	int kx = blockDim.x * blockIdx.x + threadIdx.x;
	int ky = blockDim.y * blockIdx.y + threadIdx.y;
	if (kx == 0 && ky == 0) return;
	if (kx >= hnx || ky >= ny) return;
	size_t inc = tPitch / sizeof(cuRPCF::complex) * (ky * hnx + kx);
	cuRPCF::complex* rhs_v = _rhs_v + inc;
	cuRPCF::complex* nonlinear_v = _nonlinear_v + inc;
	cuRPCF::complex* nonlinear_v_p = _nonlinear_v_p + inc;
	cuRPCF::complex* rhs_v_p = _rhs_v_p + inc;

	REAL ialpha = (REAL)kx / aphi;
	REAL ibeta = (REAL)ky / beta;
	if (ky >= ny / 2 + 1) {
		ibeta = REAL(ky - ny) / beta;
	}
	REAL kmn = ialpha * ialpha + ibeta * ibeta;

	_get_linear_v_device(rhs_v, nonlinear_v, nonlinear_v_p, rhs_v_p, nz - 1, _U0, _ddU0,
		T0, T2, T4, Re, dt, kmn, ialpha);
}