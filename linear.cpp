#include "linear.h"
#include <cassert>

int get_linear_v(problem & pb)
{
	for (int i = 0; i < (pb.mx / 2 + 1); i++) {
		for (int j = 0; j < pb.my; j++) {
			if (i == 0 && j == 0) {
				_get_linear_u0(pb.rhs_v, pb.lambx0, pb.lambx0_p, pb.nz, pb.T0, pb.T2, pb.Re, pb.dt);
				_get_linear_w0(pb.rhs_omega_y, pb.lambz0, pb.lambz0_p, pb.nz, pb.T0, pb.T2, pb.Re, pb.dt);
				continue;
			}
			size_t inc = pb.tPitch/sizeof(complex)*(j*(pb.mx/2+1)+i);
			complex* rhs_v = pb.rhs_v + inc;
			complex* nonlinear_v = pb.nonlinear_v + inc;
			complex* rhs_v_p = pb.rhs_v_p + inc;
			complex* rhs_omega_y = pb.rhs_omega_y + inc;
			complex* nonlinear_omega_y = pb.nonlinear_omega_y + inc;
			real ialpha = (real)i / pb.aphi;
			real ibeta = (real)j / pb.beta;
			if(j>= pb.my / 2 + 1) {
				ibeta = real(j - pb.my) / pb.beta;
			}
			real kmn = ialpha*ialpha + ibeta*ibeta;

			_get_linear_v(rhs_v, nonlinear_v, rhs_v_p, pb.nz, pb._U0, pb._ddU0,
				pb.T0, pb.T2, pb.T4, pb.Re, pb.dt, kmn, ialpha);
			_get_linear_omega_y(rhs_omega_y, nonlinear_omega_y, rhs_v, rhs_v_p,
				pb.nz, pb._U0, pb._dU0,
				pb.T0, pb.T2, pb.Re, pb.dt, kmn, ialpha, ibeta);
			
		}
	}
	return 0;
}

int get_linear_omega_y(problem& pb)
{
	for (int i = 0; i < (pb.mx / 2 + 1); i++) {
		for (int j = 0; j < pb.my; j++) {
			if (i == 0 && j == 0) continue;
			size_t inc = pb.tPitch / sizeof(complex)*(j*(pb.mx / 2 + 1) + i);
			complex* rhs_v = pb.rhs_v + inc;
			complex* nonlinear_v = pb.nonlinear_v + inc;
			complex* rhs_v_p = pb.rhs_v_p + inc;
			complex* rhs_omega_y = pb.rhs_omega_y + inc;
			complex* nonlinear_omega_y = pb.nonlinear_omega_y + inc;
			real ialpha = (real)i / pb.aphi;
			real ibeta = (real)j / pb.beta;
			if (j >= pb.my / 2 + 1) {
				ibeta = real(j - pb.my) / pb.beta;
			}
			real kmn = ialpha*ialpha + ibeta*ibeta;

			_get_linear_omega_y(rhs_omega_y, nonlinear_omega_y, rhs_v, rhs_v_p,
				pb.nz, pb._U0, pb._dU0,
				pb.T0, pb.T2, pb.Re, pb.dt, kmn, ialpha, ibeta);
		}
	}
	return 0;
}

int _get_linear_v(complex* rhs_v, complex* nonlinear_v, complex* rhs_v_p,
				int N, real* U, real* ddU,
				real* T0, real* T2, real* T4,
				real Re, real dt, real kmn, real alpha)
{
	complex* rhs_temp = (complex*)malloc((N+1)*sizeof(complex));
	assert(rhs_temp != nullptr);
	for (int i = 4; i <= N; i++) {
		rhs_temp[i] = complex(0.0,0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_2_0 = (N + 1)*(i-2) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_v[j] * complex(
				T4[inc_2_0]*dt*0.5/Re+(1-kmn*dt/Re)*T2[inc_2_0]
				-kmn*(1-kmn*dt*0.5/Re)*T0[inc_2_0]
				,
				-alpha*dt*0.5*U[i-2]*T2[inc_2_0]
				+(kmn*alpha*dt*0.5*U[i-2]+alpha*dt*0.5*ddU[i-2])*T0[inc_2_0]
			);
		}
	}

	//save new rhs data and add nonlinear part to it.
	for (int i = 0; i <= N; i++) {
		rhs_v_p[i] = rhs_v[i];	// save previous step v_hat data
		rhs_v[i] = rhs_temp[i] + nonlinear_v[i];
	}

	//boundary conditions
	rhs_v[0] = complex(0.0, 0.0);
	rhs_v[1] = complex(0.0, 0.0);
	rhs_v[2] = complex(0.0, 0.0);
	rhs_v[3] = complex(0.0, 0.0);

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_omega_y(complex* rhs_omega_y, complex* nonlinear_omega_y,
	complex* rhs_v, complex* rhs_v_p,
	int N, real*U, real*dU,
	real*T0, real*T2,
	real Re, real dt, real kmn,
	real alpha, real beta)
{
	complex* rhs_temp = (complex*)malloc((N + 1) * sizeof(complex));
	assert(rhs_temp != nullptr);
	for (int i = 2; i <= N; i++) {
		rhs_temp[i] = complex(0.0, 0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_1_0 = (N + 1)*(i-1) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_omega_y[j] * complex(
				T2[inc_1_0]*0.5/Re*dt + (1-kmn*dt*0.5/Re)*T0[inc_1_0]
				,
				-alpha*dt*0.5*U[i-1]*T0[inc_1_0]
			);
			rhs_temp[i] = rhs_temp[i] + (rhs_v_p[i]+rhs_v[i]) * complex(
				0.0
				,
				-0.5*beta*dU[i-1]*dt*T0[inc_1_0]
			);
		}
	}

	

	//save new rhs data and add nonlinear part to it.
	for (int i = 0; i <= N; i++) {
		rhs_omega_y[i] = rhs_temp[i] + nonlinear_omega_y[i];
	}
	
	//boundary conditions
	rhs_omega_y[0] = complex(0.0, 0.0);
	rhs_omega_y[1] = complex(0.0, 0.0);

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_u0(complex* rhs_u0, complex* lambx0, complex* lambx0_p, 
	int N, real* T0, real* T2, real Re, real dt) 
{
	complex* rhs_temp = (complex*)malloc((N + 1) * sizeof(complex));
	assert(rhs_temp != nullptr);
	for (int i = 2; i <= N; i++) {
		rhs_temp[i] = complex(0.0, 0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_1_0 = (N + 1)*(i - 1) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_u0[j] * complex(
				T0[inc_1_0] + dt*0.5 / Re*T2[inc_1_0], 0);
		}
	}


	//save new rhs data and add nonlinear part to it.
	for (int i = 2; i <= N; i++) {
		rhs_u0[i] = rhs_temp[i] - (lambx0[i]*1.5 - lambx0_p[i]*0.5);
	}

	//boundary conditions
	rhs_u0[0] = complex(0.0, 0.0);
	rhs_u0[1] = complex(0.0, 0.0);

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_w0(complex* rhs_w0, complex* lambz0, complex* lambz0_p,
	int N, real* T0, real* T2, real Re, real dt) {
	return _get_linear_u0(rhs_w0, lambz0, lambz0_p,N, T0, T2, Re, dt);
}
