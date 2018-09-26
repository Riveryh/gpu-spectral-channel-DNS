#include "velocity.h"
#include "cuRPCF.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

__global__ void getVelocityKernel(
	complex* u, complex* v, complex*w,
	complex* ox, complex*oy, complex* oz,
	int tPitch, int mx, int my, int mz, real alpha, real beta);
void get_velocity_zero(problem& pb);

int getUVW(problem& pb) {
	size_t tSize = pb.tSize;// pb.tPitch*(pb.mx / 2 + 1)*pb.my;
	
	cudaExtent tExtent = pb.tExtent;
	//make_cudaExtent(
	//	pb.mz * sizeof(complex), pb.mx/2+1, pb.my);

	cuCheck(cudaMalloc3D(&(pb.dptr_tu), tExtent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_tv), tExtent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_tw), tExtent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_tomega_x), tExtent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_tomega_y), tExtent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_tomega_z), tExtent), "allocate");

	cuCheck(cudaMemcpy(pb.dptr_tw.ptr, pb.rhs_v, tSize, cudaMemcpyHostToDevice),"cpy");
	cuCheck(cudaMemcpy(pb.dptr_tomega_z.ptr, pb.rhs_omega_y, tSize, cudaMemcpyHostToDevice), "cpy");

	//int nthreadx = 16;
	//int nthready = 16;
	//int nDimx = (pb.mx / 2 + 1) / nthreadx;
	//int nDimy = pb.my / nthready;
	//if ((pb.mx / 2 + 1) % nthreadx != 0) nDimx++;
	//if (pb.my % nthready != 0) nDimy++;
	//dim3 nThread(nthreadx, nthready);
	//dim3 nDim(nDimx, nDimy);

	getVelocityKernel <<<pb.ntDim,pb.nThread>>>((complex*)pb.dptr_tu.ptr, (complex*)pb.dptr_tv.ptr,
		(complex*)pb.dptr_tw.ptr, (complex*)pb.dptr_tomega_x.ptr, 
		(complex*)pb.dptr_tomega_y.ptr, (complex*)pb.dptr_tomega_z.ptr,
		pb.tPitch, pb.mx, pb.my, pb.mz, pb.aphi, pb.beta);
	cuCheck(cudaDeviceSynchronize(), "get velocity kernel");

	//the zeros wave number velocity 
	// is computed inside the kernel function above.
	//get_velocity_zero(pb);

	return 0;
}

__global__ void getVelocityKernel(
	complex* u, complex* v, complex*w,
	complex* ox, complex* oy, complex* oz,
	int tPitch, int mx, int my, int mz, real alpha, real beta)
{
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	int ky = threadIdx.y + blockDim.y*blockIdx.y;
	int pz = mz / 2 + 1;
	int nz = mz / 4 + 1;

	complex tdz[MAX_NZ];
	complex tdz1[MAX_NZ];

	if (kx >= (mx / 2 + 1) || ky >= my) return;

	real ialpha = real(kx) / alpha;
	real ibeta = real(ky) / beta;
	if (ky >= my / 2 + 1) {
		ibeta = real(ky - my) / beta;
	}

	real kmn = ialpha*ialpha + ibeta*ibeta;
	
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
			oy[i] = tdz[i]*(-1.0);
			oz[i] = 0.0;
		}
		return;
	}

	real kmn1 = 1.0 / kmn;

	size_t dist = (kx + (mx / 2 + 1)*ky)*tPitch/ sizeof(complex);
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
		u[i] = complex(0.0, ialpha*kmn1) * tdz[i]
			- complex(0.0, ibeta*kmn1) * oz[i];
		w[i] = complex(0.0, ibeta*kmn1) * tdz[i]
			- complex(0.0, ialpha*kmn1) * oz[i];
	}

	for (int i = 0; i < nz; i++) {
		tdz[i] = v[i];
	}
	ddz(tdz, nz);
	for (int i = 0; i < nz; i++) {
		ox[i] = tdz[i] + complex(0.0, -1.0*beta)*w[i];
	}
	for (int i = 0; i < nz; i++) {
		tdz[i] = u[i];
	}
	ddz(tdz, nz);
	for (int i = 0; i < nz; i++) {
		oy[i] = tdz[i]*(-1.0) + complex(0.0, alpha)*w[i];
	}

}

void get_velocity_zero(problem & pb)
{
	complex* w = pb.rhs_omega_y;
	complex* u = pb.rhs_v;	
}
