#pragma once
#include "data.h"

int get_linear_v(problem&pb);
int get_linear_omega_y(problem&pb);
void get_linear_zero_wave_u_w(problem& pb);

int _get_linear_v(complex* rhs_v, complex* nonlinear_v, complex* nonlinear_v_p,
	complex* rhs_v_p,
	int N, real* U, real* ddU,
	real* T0, real* T2, real* T4,
	real Re, real dt, real kmn, real alpha);

int _get_linear_omega_y(complex* rhs_omega_y, complex* nonlinear_omega_y, complex* nonlinear_omega_y_p,
				complex* rhs_v, complex* rhs_v_p,
				int N, real*U, real*dU, 
				real*T0, real*T2, 
				real Re, real dt, real kmn, 
				real alpha, real beta);

int _get_linear_u0(complex* rhs_u0, complex* lambx0, complex* lambx0_p,
	int N, real* T0, real* T2, real Re, real dt);
int _get_linear_w0(complex* rhs_w0, complex* lambz0, complex* lambz0_p,
	int N, real* T0, real* T2, real Re, real dt);