#include <iostream>
#include <time.h>  
#include <cassert>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "../include/rhs.cuh"
#include "../include/nonlinear.cuh"
#include "../include/linear.h"
#include "../include/cuRPCF.h"
#include "../include/transform.cuh"


bool subthread_inited = false;

std::condition_variable condition_v;
std::condition_variable condition_nonlinear;
std::condition_variable condition_malloc; 
std::mutex m_v;
std::mutex m_nonlinear;
std::mutex m_malloc;
std::thread sub_thread;

bool Is_malloc_completed = false;

cudaEvent_t start_non, end_non;

void save_zero_wave_number_lamb(problem& pb);
void* func(void* _pb);

void launch_subthread(problem& pb) {
	if (subthread_inited == false) {
		//pthread_cond_init(&cond_v, NULL);
		//pthread_cond_init(&cond_nonlinear, NULL);
		//pthread_cond_init(&cond_malloc, NULL);
		//pthread_mutex_init(&mutex_nonlinear, NULL);
		//pthread_mutex_init(&mutex_v, NULL);
		//pthread_mutex_init(&mutex_malloc, NULL);
		////pthread_mutex_lock(&mutex_nonlinear);
		////pthread_mutex_lock(&mutex_v);
		//pthread_create(&pid, NULL, func, &pb);
		
		sub_thread = std::thread(func, &pb);

		subthread_inited = true;
		cudaEventCreate(&start_non);
		cudaEventCreate(&end_non);
	}
}


void synchronizeGPUsolver() {
	//pthread_mutex_lock(&mutex_malloc);
	/*pthread_cond_wait(&cond_malloc, &mutex_malloc);
	pthread_mutex_unlock(&mutex_malloc);*/
	std::unique_lock<std::mutex> lk_malloc(m_malloc);
	if (Is_malloc_completed==true) {
	}
	else 
	{
		std::cout << "waiting for malloc finish" << std::endl;
		condition_malloc.wait(lk_malloc);
	}
	Is_malloc_completed = false;
	return;
}


// This function runs on a parallel thread for computing nonlinear term.
// The linear term will be computated by the main thread spontaneously to save time
// The synchronization is guaranteed by a pthread lock.
void* func(void* _pb) {
	std::cout << "sub thread launched!" << std::endl;
	std::unique_lock<std::mutex> lk_nonlinear(m_nonlinear);
	std::unique_lock<std::mutex> lk_v(m_v);
	std::unique_lock<std::mutex> lk_malloc(m_malloc);
	lk_v.unlock();
	lk_malloc.unlock();

	while (true) {
		float time;

		//wating for starting signal of the main thread
		//pthread_mutex_lock(&mutex_nonlinear);
		//pthread_cond_wait(&cond_nonlinear,&mutex_nonlinear);
		//pthread_mutex_unlock(&mutex_nonlinear);

		std::cout << "Waiting for main nonlinear" << std::endl;
		condition_nonlinear.wait(lk_nonlinear);
		
#ifdef CURPCF_CUDA_PROFILING
		clock_t start_time, end_time;
		double cost;
		start_time = clock();
		cudaEventRecord(start_non, 0);
#endif

		problem& pb = *((problem*)_pb);
		getNonlinear(pb);

#ifdef CURPCF_CUDA_PROFILING
		cudaEventRecord(end_non, 0);
		cudaEventSynchronize(end_non);
		cudaEventElapsedTime(&time, start_non, end_non);
		std::cout << "get nonlinear time CUDA EVENT = " << time / 1000.0 << std::endl;


		cudaEventRecord(start_non, 0);
#endif 

		// transform the nonlinear term into physical space.
		cheby_s2p(pb.dptr_tLamb_x, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);
		cheby_s2p(pb.dptr_tLamb_y, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);

#ifdef CURPCF_CUDA_PROFILING
		cudaEventRecord(end_non, 0);
		cudaEventSynchronize(end_non);
		cudaEventElapsedTime(&time, start_non, end_non);
		std::cout << "cheby_s2p no pad time = " << time / 1000.0 << std::endl;
#endif

		//save previous step
		cuRPCF::swap(pb.nonlinear_omega_y, pb.nonlinear_omega_y_p);
		cuRPCF::swap(pb.nonlinear_v, pb.nonlinear_v_p);
		size_t tsize = pb.tSize;// pb.tPitch * (pb.mx / 2 + 1) * pb.my;

#ifdef CURPCF_CUDA_PROFILING
		cudaEventRecord(start_non, 0);
#endif

		cuCheck(cudaMemcpy(pb.nonlinear_v, pb.dptr_tLamb_x.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
		cuCheck(cudaMemcpy(pb.nonlinear_omega_y, pb.dptr_tLamb_y.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");

#ifdef CURPCF_CUDA_PROFILING
		cudaEventRecord(end_non, 0);
		cudaEventSynchronize(end_non);
		cudaEventElapsedTime(&time, start_non, end_non);
		std::cout << "memcpy time = " << time / 1000.0 << std::endl;
#endif 

		//TODO: NONLINEAR TIME SCHEME
		save_zero_wave_number_lamb(pb);

#ifdef CURPCF_CUDA_PROFILING
		end_time = clock();
		cost = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "get nonlinear time = " << cost << std::endl;
#endif

		// synchronize with main thread.
		//pthread_mutex_lock(&mutex_v);
		//pthread_cond_signal(&cond_v);
		//pthread_mutex_unlock(&mutex_v);

		lk_v.lock();
		condition_v.notify_one();
		lk_v.unlock();

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

		// Tell the main thread the computation is completed
		//pthread_mutex_lock(&mutex_malloc);
		//pthread_cond_signal(&cond_malloc);
		//pthread_mutex_unlock(&mutex_malloc);

		lk_malloc.lock();
		std::cout << "sending malloc signal" << std::endl;
		Is_malloc_completed = true;
		condition_malloc.notify_one();
		std::cout << "malloc signal sent" << std::endl;
		lk_malloc.unlock();
	}
}

__host__ int get_rhs_v(problem& pb) {	
	std::unique_lock<std::mutex> lk_nonlinear(m_nonlinear);
	std::unique_lock<std::mutex> lk_v(m_v);
	std::unique_lock<std::mutex> lk_malloc(m_malloc);

	
	condition_nonlinear.notify_one();
	lk_nonlinear.unlock();

	std::cout << "nonlinear signal sent" << std::endl;

	//pthread_mutex_lock(&mutex_nonlinear);
	//pthread_mutex_lock(&mutex_v);
	//pthread_mutex_lock(&mutex_malloc);
	//pthread_cond_signal(&cond_nonlinear);
	//pthread_mutex_unlock(&mutex_nonlinear);

	cudaEventRecord(start_non);

	get_linear_v(pb);

	cudaEventRecord(end_non);

	//waiting for the nonlinear term by GPU
	//pthread_cond_wait(&cond_v, &mutex_v);
	//pthread_mutex_unlock(&mutex_v);

	condition_v.wait(lk_v);

	//cudaEventRecord(end_non);

	for (int i = 0; i < pb.nx / 2 + 1; i++) {
		for (int j = 0; j < pb.ny; j++) {
			for (int k = 4; k < pb.nz; k++) {
				if (i == 0 && j == 0) continue;
				size_t inc = pb.tPitch/sizeof(cuRPCF::complex)*((pb.nx/2+1)*j+i)+k;
				pb.rhs_v[inc] = pb.rhs_v[inc] +(pb.nonlinear_v[inc - 2] * 1.5 - pb.nonlinear_v_p[inc - 2] * 0.5)*pb.dt;
				//pb.rhs_v[inc] = pb.rhs_v[inc] + (pb.nonlinear_v[inc - 2])*pb.dt;
				//pb.rhs_v[inc] =  pb.nonlinear_v[inc - 2];
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
	cuRPCF::swap(pb.lambx0, pb.lambx0_p);
	cuRPCF::swap(pb.lambz0, pb.lambz0_p);
	for (int i = 0; i < pb.nz; i++) {
		pb.lambx0[i] = pb.nonlinear_v[i];
		pb.lambz0[i] = pb.nonlinear_omega_y[i];
	}
}
