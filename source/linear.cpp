#include "../include/linear.h"
#include <cassert>
#include <omp.h>
#include <iostream>
using namespace std;

int get_linear_v(problem & pb)
{
	int my = pb.my;
	const int hnx = pb.nx / 2 + 1;
	const int ny = pb.ny;
	#pragma omp parallel for// firstprivate(pb,cmx,my)
	for (int i = 0; i < hnx; i++) {
		//cout << "linear v omp id:" << omp_get_thread_num() << " i=" << i << endl;
		for (int j = 0; j < ny; j++) {
			if (i == 0 && j == 0) continue;

			size_t inc = pb.tPitch/sizeof(cuRPCF::complex)*(j*hnx+i);
			cuRPCF::complex* rhs_v = pb.rhs_v + inc;
			cuRPCF::complex* nonlinear_v = pb.nonlinear_v + inc;
			cuRPCF::complex* nonlinear_v_p = pb.nonlinear_v_p + inc;
			cuRPCF::complex* rhs_v_p = pb.rhs_v_p + inc;
			//cuRPCF::complex* rhs_omega_y = pb.rhs_omega_y + inc;
			//cuRPCF::complex* nonlinear_omega_y = pb.nonlinear_omega_y + inc;
			REAL ialpha = (REAL)i / pb.aphi;
			REAL ibeta = (REAL)j / pb.beta;
			if(j>= ny / 2 + 1) {
				ibeta = REAL(j - ny) / pb.beta;
			}
			REAL kmn = ialpha*ialpha + ibeta*ibeta;

			_get_linear_v(rhs_v, nonlinear_v, nonlinear_v_p, rhs_v_p, pb.nz-1, pb._U0, pb._ddU0,
				pb.T0, pb.T2, pb.T4, pb.Re, pb.dt, kmn, ialpha);
			//_get_linear_omega_y(rhs_omega_y, nonlinear_omega_y, rhs_v, rhs_v_p,
			//	pb.nz-1, pb._U0, pb._dU0,
			//	pb.T0, pb.T2, pb.Re, pb.dt, kmn, ialpha, ibeta);
			
		}
	}
	return 0;
}

void get_linear_zero_wave_u_w(problem& pb) {
	_get_linear_u0(pb.tv0, pb.lambx0, pb.lambx0_p, pb.nz - 1, pb.T0, pb.T2, pb.Re, pb.dt);
	_get_linear_w0(pb.tomega_y_0, pb.lambz0, pb.lambz0_p, pb.nz - 1, pb.T0, pb.T2, pb.Re, pb.dt);
	for (int k = 0; k < pb.nz; k++) {
		pb.rhs_v[k] = pb.tv0[k];
		pb.rhs_omega_y[k] = pb.tomega_y_0[k];
	}
}

int get_linear_omega_y(problem& pb)
{
	//problem pb = _pb; 
	const int hnx = pb.nx / 2+ 1;
	const int ny = pb.ny;
	#pragma omp parallel for //firstprivate(cmx,my,pb)
	for (int i = 0; i < hnx; i++) {
		//cout << "linear omg omp id:" << omp_get_thread_num() << " i=" << i << endl;
		for (int j = 0; j < ny; j++) {
			if (i == 0 && j == 0) continue;
			
			size_t inc = pb.tPitch / sizeof(cuRPCF::complex)*(j*hnx + i);
			cuRPCF::complex* rhs_v = pb.rhs_v + inc;
			cuRPCF::complex* nonlinear_v = pb.nonlinear_v + inc;
			cuRPCF::complex* rhs_v_p = pb.rhs_v_p + inc;
			cuRPCF::complex* rhs_omega_y = pb.rhs_omega_y + inc;
			cuRPCF::complex* nonlinear_omega_y = pb.nonlinear_omega_y + inc;
			cuRPCF::complex* nonlinear_omega_y_p = pb.nonlinear_omega_y_p + inc;
			REAL ialpha = (REAL)i / pb.aphi;
			REAL ibeta = (REAL)j / pb.beta;
			if (j >= ny / 2 + 1) {
				ibeta = REAL(j - ny) / pb.beta;
			}
			REAL kmn = ialpha*ialpha + ibeta*ibeta;

			_get_linear_omega_y(rhs_omega_y, nonlinear_omega_y, nonlinear_omega_y_p,
				rhs_v, rhs_v_p,
				pb.nz-1, pb._U0, pb._dU0,
				pb.T0, pb.T2, pb.Re, pb.dt, kmn, ialpha, ibeta);
		}
	}
	return 0;
}

int _get_linear_v(cuRPCF::complex* rhs_v, 
	cuRPCF::complex* nonlinear_v, cuRPCF::complex* nonlinear_v_p,
	cuRPCF::complex* rhs_v_p,
				int N, REAL* U, REAL* ddU,
				REAL* T0, REAL* T2, REAL* T4,
				REAL Re, REAL dt, REAL kmn, REAL alpha)
{
	cuRPCF::complex* rhs_temp = (cuRPCF::complex*)malloc((N+1)*sizeof(cuRPCF::complex));
	assert(rhs_temp != nullptr);
	for (int i = 4; i <= N; i++) {
		rhs_temp[i] = cuRPCF::complex(0.0,0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_2_0 = (N + 1)*(i-2) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_v[j] * cuRPCF::complex(
				T4[inc_2_0]*dt*0.5/Re+(1-kmn*dt/Re)*T2[inc_2_0]
				-kmn*(1-kmn*dt*0.5/Re)*T0[inc_2_0]
				//T2[inc_2_0] - kmn*T0[inc_2_0]
				,
				-alpha*dt*0.5*U[i-2]*T2[inc_2_0]
				+(kmn*alpha*dt*0.5*U[i-2]+alpha*dt*0.5*ddU[i-2])*T0[inc_2_0]
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

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_omega_y(cuRPCF::complex* rhs_omega_y, 
	cuRPCF::complex* nonlinear_omega_y,	cuRPCF::complex* nonlinear_omega_y_p,
	cuRPCF::complex* rhs_v, cuRPCF::complex* rhs_v_p,
	int N, REAL*U, REAL*dU,
	REAL*T0, REAL*T2,
	REAL Re, REAL dt, REAL kmn,
	REAL alpha, REAL beta)
{
	cuRPCF::complex* rhs_temp = (cuRPCF::complex*)malloc((N + 1) * sizeof(cuRPCF::complex));
	assert(rhs_temp != nullptr);
	for (int i = 2; i <= N; i++) {
		rhs_temp[i] = cuRPCF::complex(0.0, 0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_1_0 = (N + 1)*(i-1) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_omega_y[j] * cuRPCF::complex(
				T2[inc_1_0]*0.5/Re*dt + (1-kmn*dt*0.5/Re)*T0[inc_1_0]
				//T2[inc_1_0] * 0.5 / Re*dt + (0 - kmn*dt*0.5 / Re)*T0[inc_1_0]
				,
				-alpha*dt*0.5*U[i-1]*T0[inc_1_0]
			);
			rhs_temp[i] = rhs_temp[i] + (rhs_v_p[j]+rhs_v[j]) * cuRPCF::complex(
				0.0
				,
				-0.5*beta*dU[i-1]*dt*T0[inc_1_0]
			);
		}
	}

	

	//save new rhs data and add nonlinear part to it.
	for (int i = 2; i <= N; i++) {
		rhs_omega_y[i] = rhs_temp[i] + 
			(nonlinear_omega_y[i-1]*1.5 -
				nonlinear_omega_y_p[i-1]*0.5)*dt;
	}
	
	//boundary conditions
	rhs_omega_y[0] = cuRPCF::complex(0.0, 0.0);
	rhs_omega_y[1] = cuRPCF::complex(0.0, 0.0);

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_u0(cuRPCF::complex* rhs_u0, cuRPCF::complex* lambx0, cuRPCF::complex* lambx0_p, 
	int N, REAL* T0, REAL* T2, REAL Re, REAL dt) 
{
	cuRPCF::complex* rhs_temp = (cuRPCF::complex*)malloc((N + 1) * sizeof(cuRPCF::complex));
	assert(rhs_temp != nullptr);
	for (int i = 2; i <= N; i++) {
		rhs_temp[i] = cuRPCF::complex(0.0, 0.0);
		for (int j = 0; j <= N; j++) {
			size_t inc_1_0 = (N + 1)*(i - 1) + j;
			rhs_temp[i] = rhs_temp[i] + rhs_u0[j] * cuRPCF::complex(
				T0[inc_1_0] + dt*0.5 / Re*T2[inc_1_0], 0.0);
		}
	}


	//save new rhs data and add nonlinear part to it.
	for (int i = 2; i <= N; i++) {
		rhs_u0[i] = rhs_temp[i] - (lambx0[i-1] * 1.5 - lambx0_p[i-1] * 0.5) * dt;
	}

	//boundary conditions
	rhs_u0[0] = cuRPCF::complex(0.0, 0.0);
	rhs_u0[1] = cuRPCF::complex(0.0, 0.0);

	//remove pointer
	free(rhs_temp);
	return 0;
}

int _get_linear_w0(cuRPCF::complex* rhs_w0, cuRPCF::complex* lambz0, cuRPCF::complex* lambz0_p,
	int N, REAL* T0, REAL* T2, REAL Re, REAL dt) {
	return _get_linear_u0(rhs_w0, lambz0, lambz0_p,N, T0, T2, Re, dt);
}
