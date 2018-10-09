#include "rhs.cuh"
#include "nonlinear.cuh"
#include "linear.h"
#include "cuRPCF.h"
#include "transform.cuh"
#include <pthread.h>

pthread_cond_t cond_v;
pthread_mutex_t mutex_v;
pthread_t pid;
bool pthread_inited = false;

void save_zero_wave_number_lamb(problem& pb);


void* func(void* _pb) {
	problem& pb = *((problem*)_pb);
	transform(BACKWARD, pb);
	getNonlinear(pb);

	// transform the nonlinear term into physical space.
	cheby_s2p(pb.dptr_tLamb_x, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);
	cheby_s2p(pb.dptr_tLamb_y, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);

	//save previous step
	swap(pb.nonlinear_omega_y, pb.nonlinear_omega_y_p);
	swap(pb.nonlinear_v, pb.nonlinear_v_p);
	size_t tsize = pb.tSize;// pb.tPitch * (pb.mx / 2 + 1) * pb.my;
	cuCheck(cudaMemcpy(pb.nonlinear_v, pb.dptr_tLamb_x.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.nonlinear_omega_y, pb.dptr_tLamb_y.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	//TODO: NONLINEAR TIME SCHEME
	save_zero_wave_number_lamb(pb);

	// synchronize with main thread.
	pthread_mutex_lock(&mutex_v);
	pthread_cond_signal(&cond_v);
	pthread_mutex_unlock(&mutex_v);
	pthread_exit(NULL);
	return NULL;
}

__host__ int get_rhs_v(problem& pb) {
	if (pthread_inited == false) {
		pthread_cond_init(&cond_v, NULL);
		pthread_mutex_init(&mutex_v, NULL);
		pthread_inited = true;
	}

	pthread_mutex_lock(&mutex_v);
	pthread_create(&pid, NULL, func, &pb);
	
	get_linear_v(pb);

	pthread_cond_wait(&cond_v, &mutex_v);
	pthread_mutex_unlock(&mutex_v);

	for (int i = 0; i < pb.mx / 2 + 1; i++) {
		for (int j = 0; j < pb.my; j++) {
			for (int k = 4; k < pb.nz; k++) {
				if (i == 0 && j == 0) continue;
				size_t inc = pb.tPitch/sizeof(complex)*((pb.mx/2+1)*j+i)+k;
				pb.rhs_v[inc] = pb.rhs_v[inc] + (pb.nonlinear_v[inc-2]*1.5 - pb.nonlinear_v_p[inc-2]*0.5)*pb.dt;
			}
		}
	}

	get_linear_zero_wave_u_w(pb);
	return 0;

}

__host__ int get_rhs_omega(problem& pb) {
	get_linear_omega_y(pb);
	safeCudaFree(pb.dptr_tLamb_x.ptr);
	safeCudaFree(pb.dptr_tLamb_y.ptr);
	safeCudaFree(pb.dptr_tLamb_z.ptr);
	return 0;
}

void save_zero_wave_number_lamb(problem& pb) {
	swap(pb.lambx0, pb.lambx0_p);
	swap(pb.lambz0, pb.lambz0_p);
	for (int i = 0; i < pb.nz; i++) {
		pb.lambx0[i] = pb.nonlinear_v[i];
		pb.lambz0[i] = pb.nonlinear_omega_y[i];
	}
}
