#include "../include/velocity.h"
#include "../include/cuRPCF.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include <cassert>
#include <iostream>

__global__ void getVelocityKernel(
	cuRPCF::complex* u, cuRPCF::complex* v, cuRPCF::complex*w,
	cuRPCF::complex* ox, cuRPCF::complex*oy, cuRPCF::complex* oz,
	int tPitch, int mx, int my, int mz, REAL alpha, REAL beta);
__global__ void getVelocityKernel_sm(
	cuRPCF::complex* u, cuRPCF::complex* v, cuRPCF::complex*w,
	cuRPCF::complex* ox, cuRPCF::complex* oy, cuRPCF::complex* oz,
	int tPitch, int mx, int my, int mz, REAL alpha, REAL beta);
void get_velocity_zero(problem& pb);

extern cudaEvent_t __start, __stop;
extern bool cudaTimeInitialized;

int getUVW(problem& pb) {
	size_t tSize = pb.tSize;// pb.tPitch*(pb.mx / 2 + 1)*pb.my;
	float time;
//	cudaExtent tExtent = pb.tExtent;
	//make_cudaExtent(
	//	pb.mz * sizeof(cuRPCF::complex), pb.mx/2+1, pb.my);
	//cuCheck(cudaDeviceReset(),"reset");
	ASSERT(pb.dptr_tu.ptr != nullptr);
	ASSERT(pb.dptr_tv.ptr != nullptr);
	ASSERT(pb.dptr_tw.ptr != nullptr);
	ASSERT(pb.dptr_tomega_x.ptr != nullptr);
	ASSERT(pb.dptr_tomega_y.ptr != nullptr);
	ASSERT(pb.dptr_tomega_z.ptr != nullptr);

	//cuCheck(cudaMalloc3D(&(pb.dptr_tu), tExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_tv), tExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_tw), tExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_x), tExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_y), tExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_tomega_z), tExtent), "allocate");

	/*pb.dptr_tv = pb.dptr_tu; pb.dptr_tv.ptr = (char*)pb.dptr_tv.ptr + pb.size;
	pb.dptr_tw = pb.dptr_tu; pb.dptr_tw.ptr = (char*)pb.dptr_tw.ptr + pb.size;
	pb.dptr_tomega_x = pb.dptr_tu; pb.dptr_tomega_x.ptr = (char*)pb.dptr_tomega_x.ptr + pb.size;
	pb.dptr_tomega_y = pb.dptr_tu; pb.dptr_tomega_y.ptr = (char*)pb.dptr_tomega_x.ptr + pb.size;
	pb.dptr_tomega_z = pb.dptr_tu; pb.dptr_tomega_z.ptr = (char*)pb.dptr_tomega_x.ptr + pb.size;*/

	// [!!!] Note: the copy operation needs to be done before call getUVW()
	//cuCheck(cudaMemcpy(pb.dptr_tw.ptr, pb.rhs_v, tSize, cudaMemcpyHostToDevice),"cpy");
	//cuCheck(cudaMemcpy(pb.dptr_tomega_z.ptr, pb.rhs_omega_y, tSize, cudaMemcpyHostToDevice), "cpy");
	//cuCheck(cudaDeviceSynchronize(),"Mem copy");

	int nthreadx = pb.nz;
	//int nthready = 16;
	int nDimx = (pb.nx / 2 + 1);
	int nDimy = pb.ny;
	//if ((pb.mx / 2 + 1) % nthreadx != 0) nDimx++;
	//if (pb.my % nthready != 0) nDimy++;
	//dim3 nThread(nthreadx, nthready);
	dim3 nDim(nDimx, nDimy);

	if (!cudaTimeInitialized) {
		cudaEventCreate(&__start);
		cudaEventCreate(&__stop);
		cudaTimeInitialized = true;
	}

	//getVelocityKernel <<<pb.ntDim,pb.nThread>>>(
	cudaEventRecord(__start, 0);
	
	getVelocityKernel_sm <<<nDim,nthreadx>>>(
		(cuRPCF::complex*)pb.dptr_tu.ptr, (cuRPCF::complex*)pb.dptr_tv.ptr,
		(cuRPCF::complex*)pb.dptr_tw.ptr, (cuRPCF::complex*)pb.dptr_tomega_x.ptr, 
		(cuRPCF::complex*)pb.dptr_tomega_y.ptr, (cuRPCF::complex*)pb.dptr_tomega_z.ptr,
		pb.tPitch, pb.mx, pb.my, pb.mz, pb.aphi, pb.beta);
	cuCheck(cudaDeviceSynchronize(), "get velocity kernel");
	
	cudaEventRecord(__stop, 0);
	cudaEventSynchronize(__stop);
	cudaEventElapsedTime(&time, __start, __stop);
	std::cout << "get velocity time = " << time / 1000.0 << std::endl;
	//the zeros wave number velocity 
	// is computed inside the kernel function above.
	//get_velocity_zero(pb);

	return 0;
}

__global__ void getVelocityKernel_sm(
	cuRPCF::complex* u, cuRPCF::complex* v, cuRPCF::complex*w,
	cuRPCF::complex* ox, cuRPCF::complex* oy, cuRPCF::complex* oz,
	int tPitch, int mx, int my, int mz, REAL alpha, REAL beta)
{
	const int kz = threadIdx.x;
	const int kx = blockIdx.x;
	const int ky = blockIdx.y;
//	const int pz = mz / 2 + 1;
	const int nz = mz / 4 + 1;

	__shared__ cuRPCF::complex tdz[MAX_NZ];
	__shared__ cuRPCF::complex tdz1[MAX_NZ];
	
	if (kz >= nz)return;
	//solve the zero wave number case.
	if (kx == 0 && ky == 0) {
		
		u[kz] = w[kz];
		v[kz] = oz[kz];
		w[kz] = 0.0;

		tdz[kz] = u[kz];
		tdz1[kz] = v[kz];

		ddz_sm(tdz, nz, kz);
		ddz_sm(tdz1, nz, kz);
		

		ox[kz] = tdz1[kz];
		oy[kz] = tdz[kz]*(-1.0);
		oz[kz] = 0.0;

		return;
	}


	//skip empty wave numbers
	const int nx = mx / 3 * 2;
	const int ny = my / 3 * 2;
	// if (kx >= (nx / 2 + 1) || ky >= ny) return;

	REAL ialpha = REAL(kx) / alpha;
	REAL ibeta = REAL(ky) / beta;
	if (ky >= ny / 2 + 1) {
		ibeta = REAL(ky - ny) / beta;
	}

	const int i = kz;

	REAL kmn = ialpha*ialpha + ibeta*ibeta;
	REAL kmn1 = 1.0 / kmn;

	size_t dist = (kx + (nx / 2 + 1)*ky)*tPitch/ sizeof(cuRPCF::complex);
	u = u + dist;
	v = v + dist;
	w = w + dist;
	ox = ox + dist;
	oy = oy + dist;
	oz = oz + dist;

	tdz[kz] = w[kz];
	ddz_sm(tdz, nz, kz);

	u[i] = cuRPCF::complex(0.0, ialpha*kmn1) * tdz[i]
		- cuRPCF::complex(0.0, ibeta*kmn1) * oz[i];
	v[i] = cuRPCF::complex(0.0, ibeta*kmn1) * tdz[i]
	+ cuRPCF::complex(0.0, ialpha*kmn1) * oz[i];
	//u[i] = w[i];
	//v[i] = w[i];

	tdz[i] = v[i];
	ddz_sm(tdz, nz, kz);
	
	ox[i] = tdz[i] + cuRPCF::complex(0.0, -1.0*ibeta)*w[i];

	tdz[i] = u[i];
	ddz_sm(tdz, nz, kz);

	oy[i] = tdz[i]*(-1.0) + cuRPCF::complex(0.0, ialpha)*w[i];
}


__global__ void getVelocityKernel(
	cuRPCF::complex* u, cuRPCF::complex* v, cuRPCF::complex*w,
	cuRPCF::complex* ox, cuRPCF::complex* oy, cuRPCF::complex* oz,
	int tPitch, int mx, int my, int mz, REAL alpha, REAL beta)
{
	const int kx = threadIdx.x + blockDim.x*blockIdx.x;
	const int ky = threadIdx.y + blockDim.y*blockIdx.y;
	//	const int pz = mz / 2 + 1;
	const int nz = mz / 4 + 1;

	cuRPCF::complex tdz[MAX_NZ];
	cuRPCF::complex tdz1[MAX_NZ];

	//solve the zero wave number case.
	if (kx == 0 && ky == 0) {
		for (int i = 0; i < nz; i++) {
			u[i] = w[i];
			v[i] = oz[i];
			w[i] = 0.0;
			tdz[i] = u[i];
			tdz1[i] = v[i];
		}
		ddz(tdz, nz);
		ddz(tdz1, nz);
		for (int i = 0; i < nz; i++) {
			ox[i] = tdz1[i];
			oy[i] = tdz[i] * (-1.0);
			oz[i] = 0.0;
		}
		return;
	}


	//skip empty wave numbers
	const int nx = mx / 3 * 2;
	const int ny = mx / 3 * 2;
	if (kx >= (nx / 2 + 1) || ky >= ny) return;

	REAL ialpha = REAL(kx) / alpha;
	REAL ibeta = REAL(ky) / beta;
	if (ky >= ny / 2 + 1) {
		ibeta = REAL(ky - ny) / beta;
	}

	REAL kmn = ialpha*ialpha + ibeta*ibeta;
	REAL kmn1 = 1.0 / kmn;

	size_t dist = (kx + (nx / 2 + 1)*ky)*tPitch / sizeof(cuRPCF::complex);
	u = u + dist;
	v = v + dist;
	w = w + dist;
	ox = ox + dist;
	oy = oy + dist;
	oz = oz + dist;

	for (int i = 0; i < nz; i++) {
		tdz[i] = w[i];
	}

	ddz(tdz, nz);

	for (int i = 0; i < nz; i++) {
		u[i] = cuRPCF::complex(0.0, ialpha*kmn1) * tdz[i]
			- cuRPCF::complex(0.0, ibeta*kmn1) * oz[i];
		v[i] = cuRPCF::complex(0.0, ibeta*kmn1) * tdz[i]
			+ cuRPCF::complex(0.0, ialpha*kmn1) * oz[i];
		//u[i] = w[i];
		//v[i] = w[i];
	}

	for (int i = 0; i < nz; i++) {
		tdz[i] = v[i];
	}
	ddz(tdz, nz);
	for (int i = 0; i < nz; i++) {
		ox[i] = tdz[i] + cuRPCF::complex(0.0, -1.0*ibeta)*w[i];
	}
	for (int i = 0; i < nz; i++) {
		tdz[i] = u[i];
	}
	ddz(tdz, nz);
	for (int i = 0; i < nz; i++) {
		oy[i] = tdz[i] * (-1.0) + cuRPCF::complex(0.0, ialpha)*w[i];
	}
}

void get_velocity_zero(problem & pb)
{
//	cuRPCF::complex* w = pb.rhs_omega_y;
//	cuRPCF::complex* u = pb.rhs_v;	
}
