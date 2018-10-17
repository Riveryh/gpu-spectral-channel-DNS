#include "rhs.cuh"
#include "nonlinear.cuh"
#include "linear.h"
#include "cuRPCF.h"
#include "transform.cuh"
#include <pthread.h>
#include <iostream>
#include <time.h>  
#include <cassert>

pthread_cond_t cond_v;
pthread_cond_t cond_nonlinear;
pthread_mutex_t mutex_v;
pthread_t pid;
bool pthread_inited = false;
pthread_mutex_t mutex_nonlinear;
pthread_cond_t cond_malloc;
pthread_mutex_t mutex_malloc;


cudaEvent_t start_non, end_non;

void save_zero_wave_number_lamb(problem& pb);
void* func(void* _pb);

void init_pthread(problem& pb) {
	if (pthread_inited == false) {
		pthread_cond_init(&cond_v, NULL);
		pthread_cond_init(&cond_nonlinear, NULL);
		pthread_cond_init(&cond_malloc, NULL);
		pthread_mutex_init(&mutex_nonlinear, NULL);
		pthread_mutex_init(&mutex_v, NULL);
		pthread_mutex_init(&mutex_malloc, NULL);
		//pthread_mutex_lock(&mutex_nonlinear);
		//pthread_mutex_lock(&mutex_v);
		pthread_create(&pid, NULL, func, &pb);
		pthread_inited = true;
		cudaEventCreate(&start_non);
		cudaEventCreate(&end_non);
	}
}

void* func(void* _pb) {
	while (true) {
		pthread_mutex_lock(&mutex_nonlinear);
		pthread_cond_wait(&cond_nonlinear,&mutex_nonlinear);
		pthread_mutex_unlock(&mutex_nonlinear);
		clock_t start_time, end_time;
		double cost;
		problem& pb = *((problem*)_pb);
		start_time = clock();
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

		end_time = clock();
		cost = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "get nonlinear time = " << cost << std::endl;
		// synchronize with main thread.
		pthread_mutex_lock(&mutex_v);
		pthread_cond_signal(&cond_v);
		pthread_mutex_unlock(&mutex_v);
		//pthread_exit(NULL);
		//return NULL;
//		cudaExtent tExtent = pb.tExtent;

		//safeCudaFree(pb.dptr_tLamb_x.ptr);
		//safeCudaFree(pb.dptr_tLamb_y.ptr);
		//safeCudaFree(pb.dptr_tLamb_z.ptr);
		cuCheck(myCudaFree(pb.dptr_tLamb_x, ZXY_3D),"my cuda free");
		cuCheck(myCudaFree(pb.dptr_tLamb_y, ZXY_3D), "my cuda free");
		cuCheck(myCudaFree(pb.dptr_tLamb_z, ZXY_3D), "my cuda free");

		ASSERT(pb.dptr_tu.ptr == nullptr);
		ASSERT(pb.dptr_tv.ptr == nullptr);
		ASSERT(pb.dptr_tw.ptr == nullptr);
		ASSERT(pb.dptr_tomega_x.ptr == nullptr);
		ASSERT(pb.dptr_tomega_y.ptr == nullptr);
		ASSERT(pb.dptr_tomega_z.ptr == nullptr);

		//cuCheck(cudaMalloc3D(&(pb.dptr_tu), tExtent), "allocate");
		//cuCheck(cudaMalloc3D(&(pb.dptr_tv), tExtent), "allocate");
		//cuCheck(cudaMalloc3D(&(pb.dptr_tw), tExtent), "allocate");
		//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_x), tExtent), "allocate");
		//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_y), tExtent), "allocate");
		//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_z), tExtent), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tomega_z, ZXY_3D), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tomega_y, ZXY_3D), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tomega_x, ZXY_3D), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tw, ZXY_3D), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tv, ZXY_3D), "allocate");
		cuCheck(myCudaMalloc(pb.dptr_tu, ZXY_3D), "allocate");

		pthread_mutex_lock(&mutex_malloc);
		pthread_cond_signal(&cond_malloc);
		pthread_mutex_unlock(&mutex_malloc);

	}
}

__host__ int get_rhs_v(problem& pb) {	

	pthread_mutex_lock(&mutex_nonlinear);
	pthread_mutex_lock(&mutex_v);
	pthread_mutex_lock(&mutex_malloc);
	pthread_cond_signal(&cond_nonlinear);
	pthread_mutex_unlock(&mutex_nonlinear);

	cudaEventRecord(start_non);

	get_linear_v(pb);

	cudaEventRecord(end_non);

	pthread_cond_wait(&cond_v, &mutex_v);
	pthread_mutex_unlock(&mutex_v);

	//cudaEventRecord(end_non);

	for (int i = 0; i < pb.nx / 2 + 1; i++) {
		for (int j = 0; j < pb.ny; j++) {
			for (int k = 4; k < pb.nz; k++) {
				if (i == 0 && j == 0) continue;
				size_t inc = pb.tPitch/sizeof(complex)*((pb.nx/2+1)*j+i)+k;
				pb.rhs_v[inc] = pb.rhs_v[inc] + (pb.nonlinear_v[inc-2]*1.5 - pb.nonlinear_v_p[inc-2]*0.5)*pb.dt;
			}
		}
	}

	get_linear_zero_wave_u_w(pb);
	return 0;
}

__host__ int get_rhs_omega(problem& pb) {
	get_linear_omega_y(pb);
	//safeCudaFree(pb.dptr_tLamb_x.ptr);
	//safeCudaFree(pb.dptr_tLamb_y.ptr);
	//safeCudaFree(pb.dptr_tLamb_z.ptr);
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
