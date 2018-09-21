#pragma once
#include "data.h"

int get_T_matrix(int N, real* T0, real* T2, real* T4);
int get_U(int N, real* U, real* dU, real* ddU);
int _get_coefficient_v(complex* mcv, int N, real* U, real* ddU, real* T0, real* T2,
	real* T4, real Re, real dt, real kmn, real alpha);
int _get_coefficient_omega(complex* mcv, int N, real* U, real* T0, real* T2,
	real* T4, real Re, real dt, real kmn, real alpha);
int _get_coef_u0(complex* coef_u0, int N, real* T0, real* T2, real Re, real dt);
int _get_coef_w0(complex* coef_w0, int N, real* T0, real* T2, real Re, real dt);