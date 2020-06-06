#include "data.h"
#include "cuRPCF.h"
#include "coefficient.cuh"
#include <cmath>
#include <cassert>

__host__ __device__ int _get_coefficient_v(cuRPCF::complex* mcv, int N, REAL* U, REAL* ddU, REAL* T0, REAL* T2,
	REAL* T4, REAL Re, REAL dt, REAL kmn, REAL alpha)
{
	assert(mcv != nullptr);
	assert(U != nullptr);
	assert(ddU != nullptr);
	assert(T0 != nullptr);
	assert(T2 != nullptr);
	assert(T4 != nullptr);


	for (int i = 4; i <= N; i++) {
		for (int j = 0; j <= N; j++) {
			size_t inc = (N + 1)*i + j;
			size_t inc_2_0 = (N + 1)*(i - 2) + j;
			mcv[inc].re = -T4[inc_2_0]*dt*0.5/Re + (1 + kmn*dt/Re)*T2[inc_2_0]
				- kmn*(1 + kmn*dt*0.5/Re)*T0[inc_2_0];
			mcv[inc].im = alpha*dt*U[i - 2] * 0.5*T2[inc_2_0]
				- (kmn*alpha*dt*0.5*U[i - 2] + alpha*dt*0.5*ddU[i - 2])*T0[inc_2_0];
		}
	}

	//boundary conditions
	for (int j = 0; j <= N; j++) {
		mcv[(N + 1) * 0 + j] = cuRPCF::complex(1.0, 0.0);
		mcv[(N + 1) * 1 + j] = cuRPCF::complex((j%2==0)?1.0:-1.0, 0.0);
		mcv[(N + 1) * 2 + j] = cuRPCF::complex(j*j, 0.0);
		mcv[(N + 1) * 3 + j] = cuRPCF::complex(((j%2==0) ? -1.0 : 1.0)*j*j,0.0);
	}
	return 0;
}

__host__ __device__ int _get_coefficient_omega(cuRPCF::complex* mcv, int N, REAL* U, REAL* T0, REAL* T2,
	REAL* T4, REAL Re, REAL dt, REAL kmn, REAL alpha)
{
	assert(mcv != nullptr);
	assert(U != nullptr);
	assert(T0 != nullptr);
	assert(T2 != nullptr);
	assert(T4 != nullptr);


	for (int i = 2; i <= N; i++) {
		for (int j = 0; j <= N; j++) {
			size_t inc = (N + 1)*i + j;
			size_t inc_1_0 = (N + 1)*(i - 1) + j;
			mcv[inc].re = -T2[inc_1_0] * dt*0.5/Re + (1 + kmn*dt*0.5/Re)*T0[inc_1_0];
			mcv[inc].im = alpha*dt*0.5*U[i-1] * T0[inc_1_0];
		}
	}

	//boundary conditions
	for (int j = 0; j <= N; j++) {
		mcv[(N + 1) * 0 + j] = cuRPCF::complex(1.0, 0.0);
		mcv[(N + 1) * 1 + j] = cuRPCF::complex((j%2 == 0) ? 1.0 : -1.0, 0.0);
	}
	return 0;
}

__host__ __device__ int _get_coef_u0(cuRPCF::complex* coef_u0, int N, REAL* T0, REAL* T2, REAL Re, REAL dt) {
	for (int i = 2; i <= N; i++) {
		for (int j = 0; j <= N; j++) {
			size_t inc = (N + 1)*i + j;
			size_t inc_1_0 = (N + 1)*(i-1) + j;
			coef_u0[inc] = -T2[inc_1_0] * dt*0.5 / Re + T0[inc_1_0];
		}
	}

	//boundary conditions
	for (int j = 0; j <= N; j++) {
		coef_u0[(N + 1) * 0 + j] = cuRPCF::complex(1.0, 0.0);
		coef_u0[(N + 1) * 1 + j] = cuRPCF::complex((j%2==0)?1.0:-1.0, 0.0);
	}
	return 0;
}

__host__ __device__ int _get_coef_w0(cuRPCF::complex* coef_w0, int N, REAL* T0, REAL* T2, REAL Re, REAL dt) {
	return 
		_get_coef_u0(coef_w0, N, T0, T2, Re, dt);
}

__host__ __device__ int get_T_matrix(int N, REAL* T0, REAL* T2, REAL* T4) {
	REAL PI = 4.0*atan(1.0);
	REAL* T1;
	REAL* T3;
	T1 = (REAL*)malloc((N + 1)*(N + 1) * sizeof(REAL));
	T3 = (REAL*)malloc((N + 1)*(N + 1) * sizeof(REAL));
	ASSERT(T0 != nullptr);
	ASSERT(T1 != nullptr);
	ASSERT(T2 != nullptr);
	ASSERT(T3 != nullptr);
	ASSERT(T4 != nullptr);

	// Dimension of T0/2/4 matrix N+1 * N+1
	for (int i = 0; i <= N; i++) {
		for (int j = 0; j <= N; j++) {
			size_t inc = (N + 1)*i + j;
			T0[inc] = cos(j*PI*i / N);
		}
	}

	for (int j = 0; j <= 3; j++) {
		for (int i = 0; i <= N; i++) {
			size_t inc = (N + 1)*i + j;
			T1[inc] = 0.0;
			T2[inc] = 0.0;
			T3[inc] = 0.0;
			T4[inc] = 0.0;
		}
	}

	for (int i = 0; i <= N; i++) {
		size_t inc1 = (N + 1)*i + 1;
		size_t inc0 = (N + 1)*i + 0;
		T1[inc1] = T0[inc0];
	}
	
	for (int i = 0; i <= N; i++) {
		size_t inc2 = (N + 1)*i + 2;
		size_t inc1 = (N + 1)*i + 1;
		size_t inc0 = (N + 1)*i + 0;
		T2[inc2] = 4 * T0[inc0];
		T1[inc2] = 4 * T0[inc1];
	}

	for (int j = 3; j <= N; j++) {
		for (int i = 0; i <= N; i++) {
			size_t incJ = (N + 1)*i + j;
			size_t incJ_1 = (N + 1)*i + j - 1;
			size_t incJ_2 = (N + 1)*i + j - 2;
			T1[incJ] = 2 * j*T0[incJ_1] + j*T1[incJ_2] / (j - 2);
			T2[incJ] = 2 * j*T1[incJ_1] + j*T2[incJ_2] / (j - 2);
			T3[incJ] = 2 * j*T2[incJ_1] + j*T3[incJ_2] / (j - 2);
			T4[incJ] = 2 * j*T3[incJ_1] + j*T4[incJ_2] / (j - 2);
		}
	}

	free(T1);
	free(T3);
	return 0;
}

__host__ __device__ int get_U(int N, REAL * U, REAL * dU, REAL * ddU)
{
	REAL PI = 4.0*atan(1.0);
	for (int i = 0; i < N; i++) {
		REAL z = cos((REAL)i / (N - 1)* PI);
		U[i] = 0.5*(1+z);
		dU[i] = 0.5;
		ddU[i] = 0.0;
	}
	return 0;
}
