#pragma once
#include "data.h"

__host__ __device__ int get_T_matrix(int N, REAL* T0, REAL* T2, REAL* T4);
__host__ __device__ int get_U(int N, REAL* U, REAL* dU, REAL* ddU);
__host__ __device__ int _get_coefficient_v(cuRPCF::complex* mcv, int N, REAL* U, REAL* ddU, REAL* T0, REAL* T2,
	REAL* T4, REAL Re, REAL dt, REAL kmn, REAL alpha);
__host__ __device__ int _get_coefficient_omega(cuRPCF::complex* mcv, int N, REAL* U, REAL* T0, REAL* T2,
	REAL* T4, REAL Re, REAL dt, REAL kmn, REAL alpha);
__host__ __device__ int _get_coef_u0(cuRPCF::complex* coef_u0, int N, REAL* T0, REAL* T2, REAL Re, REAL dt);
__host__ __device__ int _get_coef_w0(cuRPCF::complex* coef_w0, int N, REAL* T0, REAL* T2, REAL Re, REAL dt);