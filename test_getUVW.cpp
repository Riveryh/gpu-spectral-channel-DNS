#include "test_getUVW.h"
#include "velocity.h"
#include "operation.h"
#include "solver.h"
#include "cuRPCF.h"
#include "transform.cuh"
#include "rhs.cuh"
#include <cassert>
#include <iostream>
using namespace std;

TestResult compare_velocities_and_vortricities(problem& pb);
void _setFlow(problem& pb);

TestResult test_getUVW() {
	RPCF_Paras para("parameter.txt");
	problem pb(para);
	initCUDA(pb);
	initFFT(pb);
	initSolver(pb);
	//initFlow(pb);


	size_t size = pb.pitch * pb.my * pb.mz;
	//memory allocation
	pb.hptr_u = (real*)malloc(size);
	assert(pb.hptr_u != nullptr);
	pb.hptr_v = (real*)malloc(size);
	assert(pb.hptr_v != nullptr);
	pb.hptr_w = (real*)malloc(size);
	assert(pb.hptr_w != nullptr);
	pb.hptr_omega_x = (real*)malloc(size);
	assert(pb.hptr_omega_x != nullptr);
	pb.hptr_omega_y = (real*)malloc(size);
	assert(pb.hptr_omega_y != nullptr);
	pb.hptr_omega_z = (real*)malloc(size);
	assert(pb.hptr_omega_z != nullptr);

	_setFlow(pb);

	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	
	//get_rhs_v(pb);
	//RPCF::write_3d_to_file("rhs_v.txt", (real*)pb.rhs_v,
	//	pb.tPitch, 2*pb.mz, (pb.mx / 2 + 1), pb.my);
	pb.Ro = 0;

	cuCheck(cudaMemcpy(pb.dptr_tw.ptr, pb.rhs_v, pb.tSize, cudaMemcpyHostToDevice),"cpy");
	cuCheck(cudaMemcpy(pb.dptr_tomega_z.ptr, pb.rhs_omega_y, pb.tSize, cudaMemcpyHostToDevice), "cpy");

	getUVW(pb);

	TestResult res = compare_velocities_and_vortricities(pb);
	
	ASSERT(res == TestSuccess);

	return res;
}

TestResult compare_velocities_and_vortricities(problem& pb) {

	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pz = pb.pz;
	int pitch = pb.pitch;
	real lx = pb.lx;
	real ly = pb.ly;
	real* u = pb.hptr_u;
	real* v = pb.hptr_v;
	real* w = pb.hptr_w;
	real* ox = pb.hptr_omega_x;
	real* oy = pb.hptr_omega_y;
	real* oz = pb.hptr_omega_z;

	real _u, _v, _w, _ox, _oy, _oz;

	size_t size = pitch * my * pz;


	int dim[3];
	dim[0] = pb.mx;
	dim[1] = pb.my;
	dim[2] = pb.mz;

	int tDim[3];
	tDim[0] = pb.mz;
	tDim[1] = pb.mx;
	tDim[2] = pb.my;

	//real* hptr_tomega_x = (real*)malloc(pb.tSize);
	//real* hptr_tomega_z = (real*)malloc(pb.tSize);
	//real* hptr_tu = (real*)malloc(pb.tSize);
	//real* hptr_tv = (real*)malloc(pb.tSize);
	//real* hptr_tw = (real*)malloc(pb.tSize);
	//cuCheck(cudaMemcpy(hptr_tomega_x, pb.dptr_tomega_x.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	//cuCheck(cudaMemcpy(hptr_tomega_z, pb.dptr_tomega_z.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	//cuCheck(cudaMemcpy(hptr_tu, pb.dptr_tu.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	//cuCheck(cudaMemcpy(hptr_tv, pb.dptr_tv.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	//cuCheck(cudaMemcpy(hptr_tw, pb.dptr_tw.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");

	transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	
	transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);
	
	cuCheck(cudaMemcpy(pb.hptr_u, pb.dptr_u.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_v, pb.dptr_v.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_w, pb.dptr_w.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_x, pb.dptr_omega_x.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_y, pb.dptr_omega_y.ptr, size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_z, pb.dptr_omega_z.ptr, size, cudaMemcpyDeviceToHost), "memcpy");


	real PI = 4.0*atan(1.0);
	real pi = PI;
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < my; j++)
			for (int i = 0; i < mx; i++)
			{
				real x = lx * i / mx;
				real y = ly * j / my;
				real z = cos(real(k) / (pz - 1)*PI);
				size_t inc = (pitch * my * k + pitch *j) / sizeof(real) + i;
				_u = sin(x);
				_v = cos(y);
				_w = -z*cos(x) + z*sin(y);
				_ox = -z*cos(y);
				_oy = z*sin(x);
				_oz = 0;

				assert(isEqual(_u, u[inc]));
				assert(isEqual(_v, v[inc]));
				assert(isEqual(_w, w[inc]));
				assert(isEqual(_ox, ox[inc]));
				assert(isEqual(_oy, oy[inc]));
				assert(isEqual(_oz, oz[inc]));
			}

	return TestSuccess;
}


void _setFlow(problem& pb) {
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pz = pb.pz;
	int pitch = pb.pitch;
	real lx = pb.lx;
	real ly = pb.ly;
	real* u = pb.hptr_u;
	real* v = pb.hptr_v;
	real* w = pb.hptr_w;
	real* ox = pb.hptr_omega_x;
	real* oy = pb.hptr_omega_y;
	real* oz = pb.hptr_omega_z;
	size_t size = pitch * my * pz;

	real PI = 4.0*atan(1.0); 
	real pi = PI;
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < my; j++)
			for (int i = 0; i < mx; i++)
			{
				real x = lx * i / mx;
				real y = ly * j / my;
				real z = cos(real(k) / (pz - 1)*PI);
				size_t inc = (pitch * my * k + pitch *j) / sizeof(real) + i;
				u[inc] = sin(x);
				v[inc] = cos(y);
				w[inc] = -z*cos(x) + z*sin(y);
				ox[inc] = -z*cos(y);
				oy[inc] = z*sin(x);
				oz[inc] = 0;
			}

	cuCheck(cudaMemcpy(pb.dptr_u.ptr, pb.hptr_u, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_v.ptr, pb.hptr_v, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_w.ptr, pb.hptr_w, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_x.ptr, pb.hptr_omega_x, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_y.ptr, pb.hptr_omega_y, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_z.ptr, pb.hptr_omega_z, size, cudaMemcpyHostToDevice), "memcpy");

	int dim[3];
	dim[0] = pb.mx;
	dim[1] = pb.my;
	dim[2] = pb.mz;

	int tDim[3];
	tDim[0] = pb.mz;
	tDim[1] = pb.mx;
	tDim[2] = pb.my;

	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);

}
