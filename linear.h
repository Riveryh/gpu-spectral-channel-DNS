#pragma once
#include "data.h"

int get_linear_v(problem&pb);
int get_linear_omega_y(problem&pb);
void get_linear_zero_wave_u_w(problem& pb);

int _get_linear_v(cuRPCF::complex* rhs_v, cuRPCF::complex* nonlinear_v, cuRPCF::complex* nonlinear_v_p,
	cuRPCF::complex* rhs_v_p,
	int N, REAL* U, REAL* ddU,
	REAL* T0, REAL* T2, REAL* T4,
	REAL Re, REAL dt, REAL kmn, REAL alpha);

int _get_linear_omega_y(cuRPCF::complex* rhs_omega_y, cuRPCF::complex* nonlinear_omega_y, cuRPCF::complex* nonlinear_omega_y_p,
				cuRPCF::complex* rhs_v, cuRPCF::complex* rhs_v_p,
				int N, REAL*U, REAL*dU, 
				REAL*T0, REAL*T2, 
				REAL Re, REAL dt, REAL kmn, 
				REAL alpha, REAL beta);

int _get_linear_u0(cuRPCF::complex* rhs_u0, cuRPCF::complex* lambx0, cuRPCF::complex* lambx0_p,
	int N, REAL* T0, REAL* T2, REAL Re, REAL dt);
int _get_linear_w0(cuRPCF::complex* rhs_w0, cuRPCF::complex* lambz0, cuRPCF::complex* lambz0_p,
	int N, REAL* T0, REAL* T2, REAL Re, REAL dt);