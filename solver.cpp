#include "solver.h"
#include "rhs.cuh"
#include "malloc.h"
#include "matrix_op.h"
#include "velocity.h"
#include "coefficient.cuh"
#include "operation.h"
#include <cassert>
//// compute multiply of matrix and vector
//void multiplyMatrix(complex* mul, complex* v, const int n);
//
//// compute coeffecient matrix of v
//void getCoefV(complex * coefv, int n, real kmn, real alpha,
//	matrix2d<real>& T0, matrix2d<real>& T2, matrix2d<real>& T4,
//	real* U0, real* dU0, real* ddU0, const real dt, const real Re);
//
//// compute coeffecient matrix of omega
//void getCoefOmega(complex * coefOmega, int n, real kmn, real alpha,
//	matrix2d<real>& T0, matrix2d<real>& T2, matrix2d<real>& T4,
//	real* U0, real* dU0, real* ddU0, const real dt, const real Re);
int solveEq(complex* inv_coef, complex* rhs, int N, 
	size_t pitch, int nx, int ny);
void save_0_v_omega_y(problem& pb);

int nextStep(problem& pb) {
	get_rhs_v(pb);

	//solve equation of v from (0,0) to (nx,ny)
	solveEq(pb.matrix_coeff_v, pb.rhs_v, 
		pb.nz, pb.tPitch, pb.mx, pb.my);//check the dimension of data??

	get_rhs_omega(pb);
	//solve equation of omega from (0,0) to (nx,ny)
	solveEq(pb.matrix_coeff_omega, pb.rhs_omega_y, 
		pb.nz, pb.tPitch, pb.mx, pb.my);
	
	save_0_v_omega_y(pb);

	getUVW(pb);
	pb.currenStep++;
	return 0;
}

int startLoop(problem& pb) {
	return 0;
}

int solveEq(complex* inv_coef, complex* rhs, int N,
			size_t pitch, int mx, int my) {
	for (int i = 0; i < mx/2+1; i++) {
		for (int j = 0; j < my; j++) {
			size_t inc_m = N*N*((mx/2+1)*j + i);
			size_t inc_rhs = pitch / sizeof(complex) * ((mx/2+1)*j + i);
			m_multi_v(inv_coef + inc_m, rhs + inc_rhs, N);
		}
	}
	return 0;
}

//void multiplyMatrix(complex * mul, complex * v, const int n)
//{
//	complex* temp = (complex*)malloc(n * sizeof(complex));
//	for (int i = 0; i < n; i++) {
//		auto* ai = mul + n*i;
//		temp[i].re = 0.0;
//		temp[i].im = 0.0;
//		for (int j = 0; j < n; j++) {
//			temp[i].re = temp[i].re + ai[j].re * v[j].re - ai[j].im * v[j].im;
//			temp[i].im = temp[i].im + ai[j].im * v[j].re + ai[j].re * v[j].im;
//		}
//	}
//	for (int i = 0; i < n; i++) {
//		v[i] = temp[i];
//	}
//	free(temp);
//}
//
//void getCoefV(complex * coefv, int n, real kmn, real alpha, 
//	matrix2d<real>& T0, matrix2d<real>& T2, matrix2d<real>& T4, 
//	real* U0, real* dU0, real* ddU0 ,const real dt,const real Re)
//{
//	for (int i = 4; i < n; i++) {
//		for (int j = 0; j < n; j++) {
//			complex* coefIJ = coefv + (i*n + j);
//			coefIJ->re = -T4(i - 2, j)*dt*0.5 / Re 
//				+ (1 + kmn*dt / Re)*T2(i - 2, j)
//				- kmn*(1 + kmn*dt*0.5 / Re)*T0(i - 2, j);
//			coefIJ->im = alpha*dt*U0[i - 2] * 0.5 / Re*T2(i - 2, j)
//				- (kmn*alpha*dt*0.5*U0[i - 2]
//					- alpha*dt*0.5*ddU0[i - 2])*T0(i - 2, j);
//		}
//	}
//
//	
//	for (int j = 0; j < n; j++) {
//		complex* a0j = coefv + j;
//		complex* a1j = coefv + n + j;
//		complex* a2j = coefv + 2 * n + j;
//		complex* a3j = coefv + 3 * n + j;
//		*a0j = complex(0, 0);
//		*a1j = complex((j % 2 == 0) ? 1 : -1, 0);
//		*a2j = complex(j*j, 0);
//		*a3j = complex(((j % 2 == 0) ? -1 : 1)*j*j, 0);
//	}
//}
//
//void getCoefOmega(complex * coefOmega, int n, real kmn, real alpha,
//	matrix2d<real>& T0, matrix2d<real>& T2, matrix2d<real>& T4,
//	real* U0, real* dU0, real* ddU0, const real dt, const real Re)
//{
//	for (int i = 2; i < n; i++) {
//		for (int j = 0; j < n; j++) {
//			complex* coefIJ = coefOmega + (i*n + j);
//			coefIJ->re = -T2(i - 1, j)*dt*0.5 / Re
//				+ (1 + kmn*dt*0.5 / Re)*T0(i - 1, j);
//			coefIJ->im = alpha*dt*0.5*U0[i - 1] * T0(i - 1, j);
//		}
//	}
//	
//	for (int j = 0; j < n; j++) {
//		complex* a0j = coefOmega + j;
//		complex* a1j = coefOmega + n + j;
//		*a0j = complex(1, 0);
//		*a1j = complex((j % 2 == 0) ? 1 : -1, 0);
//	}
//}


int initSolver(problem& pb, bool inversed)
{
	const int nz = pb.nz;
	const int nx = pb.nx;
	const int ny = pb.ny;

	size_t& tSize = pb.tSize;
	size_t mSize = pb.nz * pb.nz * (pb.mx / 2 + 1)*pb.my * sizeof(complex);

	pb.matrix_coeff_v = (complex*)malloc(mSize);
	pb.matrix_coeff_omega = (complex*)malloc(mSize);

	pb.nonlinear_v = (complex*)malloc(tSize);
	pb.nonlinear_omega_y = (complex*)malloc(tSize);
	pb.nonlinear_v_p = (complex*)malloc(tSize);
	pb.nonlinear_omega_y_p = (complex*)malloc(tSize);

	pb.rhs_v = (complex*)malloc(tSize);
	pb.rhs_omega_y = (complex*)malloc(tSize);
	pb.rhs_v_p = (complex*)malloc(tSize);

	pb.lambx0 = (complex*)malloc(sizeof(complex)*pb.nz);
	pb.lambz0 = (complex*)malloc(sizeof(complex)*pb.nz); 
	pb.lambx0_p = (complex*)malloc(sizeof(complex)*pb.nz);
	pb.lambz0_p = (complex*)malloc(sizeof(complex)*pb.nz);
	pb.tv0 = (complex*)malloc(sizeof(complex)*pb.nz);
	pb.tomega_y_0 = (complex*)malloc(sizeof(complex)*pb.nz);

	pb._U0 = (real*)malloc(sizeof(real)*pb.nz);
	pb._dU0 = (real*)malloc(sizeof(real)*pb.nz);
	pb._ddU0 = (real*)malloc(sizeof(real)*pb.nz);

	pb.T0 = (real*)malloc(sizeof(real)*pb.nz*pb.nz);
	pb.T2 = (real*)malloc(sizeof(real)*pb.nz*pb.nz);
	pb.T4 = (real*)malloc(sizeof(real)*pb.nz*pb.nz);

	//init T matrix
	get_T_matrix(pb.nz-1, pb.T0, pb.T2, pb.T4);

	//init mean velocity vector
	get_U(pb.nz, pb._U0, pb._dU0, pb._ddU0);

	//init coef matrix
	for (int kx = 0; kx < (pb.mx / 2 +1); kx++) {
		for (int ky = 0; ky < pb.my; ky++) {
			
			if (kx == 0 && ky == 0) {
				_get_coef_u0(pb.matrix_coeff_v, pb.nz-1, pb.T0, pb.T2, pb.Re, pb.dt);
				_get_coef_w0(pb.matrix_coeff_omega, pb.nz-1, pb.T0, pb.T2, pb.Re, pb.dt);
				if (inversed) {
					int ret;
					ret=inverse(pb.matrix_coeff_omega, pb.nz);
					ASSERT(ret == 0);
					ret=inverse(pb.matrix_coeff_v, pb.nz);
					ASSERT(ret == 0);
				}
				continue;
			}

			real ialpha, ibeta;
			get_ialpha_ibeta(kx, ky, pb.my, pb.aphi, pb.beta, ialpha, ibeta);

			real kmn = ialpha*ialpha + ibeta*ibeta;

			size_t inc = pb.nz*pb.nz*((pb.mx/2+1)*ky + kx);
			complex* coe_v = pb.matrix_coeff_v + inc;
			complex* coe_o = pb.matrix_coeff_omega + inc;

			_get_coefficient_v(coe_v, pb.nz-1, pb._U0, pb._ddU0,
				pb.T0, pb.T2, pb.T4, pb.Re, pb.dt, kmn, ialpha);
			_get_coefficient_omega(coe_o, pb.nz-1, pb._U0, 
				pb.T0, pb.T2, pb.T4, pb.Re, pb.dt, kmn, ialpha);
			if (inversed) {
				int ret;
				ret = inverse(coe_v, pb.nz);
				ASSERT(ret == 0);
				ret = inverse(coe_o, pb.nz);
				ASSERT(ret == 0);
			}
		}
	}

	pb.currenStep = pb.para.stepPara.start_step;

	return 0;
}

int destroySolver(problem& pb) {
	free(pb.matrix_coeff_omega);
	free(pb.matrix_coeff_v);
	free(pb.nonlinear_v);
	free(pb.nonlinear_omega_y);
	free(pb.nonlinear_v_p);
	free(pb.nonlinear_omega_y_p);
	return 0;
}

void save_0_v_omega_y(problem& pb) {
	const int nz = pb.nz;
	for (int i = 0; i < pb.nz; i++) {
		pb.tv0[i] = pb.rhs_v[i];
		pb.tomega_y_0[i] = pb.rhs_omega_y[i];
	}
}

