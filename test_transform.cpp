#include "test_transform.h"
#include "cuRPCF.h"
#include <iostream>
using namespace std;
#include "test_nonlinear.h"
#include <cassert>

void compareFlow(problem& pb);
void setFlowForSpectra(problem& pb);
void compareSpectra(problem& pb);

TestResult test_transform() {
	//problem pb;
	//initcuda(pb);
	//initfft(pb);

	//int mx = pb.mx;
	//int my = pb.my;
	//int mz = pb.mz;
	//int pitch = pb.pitch;

	//size_t size = pitch * my * mz;
	////memory allocation
	//pb.hptr_u = (real*)malloc(size);
	//assert(pb.hptr_u != nullptr);
	//pb.hptr_v = (real*)malloc(size);
	//assert(pb.hptr_v != nullptr);
	//pb.hptr_w = (real*)malloc(size);
	//assert(pb.hptr_w != nullptr);
	//pb.hptr_omega_x = (real*)malloc(size);
	//assert(pb.hptr_omega_x != nullptr);
	//pb.hptr_omega_y = (real*)malloc(size);
	//assert(pb.hptr_omega_y != nullptr);
	//pb.hptr_omega_z = (real*)malloc(size);
	//assert(pb.hptr_omega_z != nullptr);

	//setflow(pb);

	//int dim[3] = { pb.mx,pb.my,pb.mz };
	//int tdim[3] = { pb.mz,pb.mx,pb.my };
	//transform_3d_one(forward, pb.dptr_u, pb.dptr_tu, dim, tdim);
	//transform_3d_one(backward, pb.dptr_u, pb.dptr_tu, dim, tdim);

	////-------------------------------
	//compareflow(pb);

	problem pb2;
	initCUDA(pb2);
	initFFT(pb2);
	int mx = pb2.mx;
	int my = pb2.my;
	int mz = pb2.mz;
	int pitch = pb2.pitch;
	size_t size2 = pitch * my * mz;
	//memory allocation
	pb2.hptr_u = (real*)malloc(size2);
	assert(pb2.hptr_u != nullptr);
	int dim2[3] = { pb2.mx,pb2.my,pb2.mz };
	int tdim2[3] = { pb2.mz,pb2.mx,pb2.my };
	setFlowForSpectra(pb2);
	transform_3d_one(FORWARD, pb2.dptr_u, pb2.dptr_tu, dim2, tdim2);
	compareSpectra(pb2);

	

	return TestSuccess;
}

void compareFlow(problem& pb) {
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pz = (mz / 2 + 1);
	size_t pitch = pb.pitch;
	real lx = pb.lx;
	real ly = pb.ly;
	real* u = pb.hptr_u;
	size_t size = pitch * my * mz;
	cuCheck(cudaMemcpy(pb.hptr_u, pb.dptr_u.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaDeviceSynchronize(),"Sync");
	real PRECISION = 1e-8;

	real PI = 4.0*atan(1.0);
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < my; j++)
			for (int i = 0; i < mx; i++)
			{
				real x = lx * i / mx;
				real y = ly * j / my;
				real z = cos(real(k) / (pz - 1)*PI);
				size_t inc = (pitch * my * k + pitch *j) / sizeof(real) + i;
				real ex_u = (1 - z*z)*sin(y)*cos(x);
				assert(isEqual(ex_u, u[inc], PRECISION));
			}
}


void compareSpectra(problem& pb) {
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pitch = pb.tPitch;
	real lx = pb.lx;
	real ly = pb.ly;
	complex* u;
	size_t size = pitch * (mx / 2 + 1)*my;
	u =(complex*)malloc(size);
	assert(u != nullptr);
	cuCheck(cudaMemcpy(u, pb.dptr_tu.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	real PRECISION = 1e-8;

	real PI = 4.0*atan(1.0);
	for (int j = 0; j < my; j++)
		for (int i = 0; i < mx/2+1; i++)
			for (int k=0; k < mz/2+1; k++)
			{
				size_t inc = pitch*((mx / 2 + 1)*j + i) / sizeof(complex) + k;
				if (i == 0 && j == 0) {
					if (k == 0) {
						assert(isEqual(u[inc].re, 0.5, PRECISION));
					}
					else if (k == 2) {
						assert(isEqual(u[inc].re, -0.5, PRECISION));
					}
					else {
						assert(isEqual(u[inc].re, 0.0, PRECISION));
					}				
				}
				else {
					assert(isEqual(u[inc].re, 0.0, PRECISION));
				}
				assert(isEqual(u[inc].im, 0.0, PRECISION));
			}
}


void setFlowForSpectra(problem& pb) {
	int px = pb.mx;
	int py = pb.my;
	int pz = pb.mz/2+1;
	int pitch = pb.pitch;
	real lx = pb.lx;
	real ly = pb.ly;
	real* u = pb.hptr_u;
	size_t size = pitch * py * pz;

	real PI = 4.0*atan(1.0);
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < py; j++)
			for (int i = 0; i < px; i++)
			{
				real x = lx * i / px;
				real y = ly * j / py;
				real z = cos(real(k) / (pz - 1)*PI);
				size_t inc = (pitch * py * k + pitch *j) / sizeof(real) + i;
				u[inc] = 1 - z*z;
			}

	cuCheck(cudaMemcpy(pb.dptr_u.ptr, pb.hptr_u, size, cudaMemcpyHostToDevice), "memcpy");
}