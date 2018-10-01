#include "test_rhs.h"
#include "data.h"
#include "operation.h"
#include "transform.cuh"
#include "RPCFKernels.cuh"
#include "solver.h"
#include "test_transform.h"
#include "test_nonlinear.h"
#include "nonlinear.cuh"
#include <cassert>
#include "cuRPCF.h"
#include "rhs.cuh"
#include <iostream>
#include "output.h"

TestResult test_rhs() {
	RPCF_Paras para("parameter.txt");
	problem pb(para);
	initCUDA(pb);
	initFFT(pb);
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pitch = pb.pitch;
	size_t size = pitch * my * mz;
	initSolver(pb);
	//set_flow_rhs_test(pb);
	initFlow(pb);
	output_velocity(pb);
	get_rhs_v(pb);
	initCUDA(pb);
	//set_flow_rhs_test(pb);
	initFlow(pb);
	get_rhs_v(pb);
	//nextStep(pb);
	//compare_rhs_v(pb);
	/*for (int i = 0; i < 10; i++) {
		std::cout << "step: " << i << std::endl;
		nextStep(pb);
		output_velocity(pb);
	}*/

	RPCF::write_3d_to_file("rhs_v.txt", (real*)pb.rhs_v,
		pb.tPitch, 2 * pb.mz, (pb.mx / 2 + 1), pb.my);
	RPCF::write_3d_to_file("nonliear_v.txt", (real*)pb.nonlinear_v,
		pb.tPitch, 2 * pb.mz, (pb.mx / 2 + 1), pb.my);
	//RPCF::write_3d_to_file("rhs_omega.txt", (real*)pb.rhs_omega_y,
	//	pb.tPitch, 2 * pb.mz, (pb.mx / 2 + 1), pb.my);
	//RPCF::write_3d_to_file("nonliear_omega.txt", (real*)pb.nonlinear_omega_y,
	//	pb.tPitch, 2 * pb.mz, (pb.mx / 2 + 1), pb.my);
	return TestSuccess;
}

void set_flow_rhs_test(problem& pb)
{
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pz = pb.pz;
	int pitch = pb.pitch;
	real lx = pb.lx;
	real ly = pb.ly;

	pb.hptr_u = (real*)malloc(pb.size);
	pb.hptr_v = (real*)malloc(pb.size);
	pb.hptr_w = (real*)malloc(pb.size);
	pb.hptr_omega_x = (real*)malloc(pb.size);
	pb.hptr_omega_y = (real*)malloc(pb.size);
	pb.hptr_omega_z = (real*)malloc(pb.size);

	real* u = pb.hptr_u;
	real* v = pb.hptr_v;
	real* w = pb.hptr_w;
	real* ox = pb.hptr_omega_x;
	real* oy = pb.hptr_omega_y;
	real* oz = pb.hptr_omega_z;

	size_t size = pitch * my * mz;

	real PI = 4.0*atan(1.0);
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < my; j++)
			for (int i = 0; i < mx; i++)
			{
				real x = lx * i / mx;
				real y = ly * j / my;
				real z = cos(real(k) / (pz - 1)*PI);
				size_t inc = (pitch * my * k + pitch *j) / sizeof(real) + i;
				u[inc] = (1 - z*z)*sin(y);
				v[inc] = 0.0;
				w[inc] = 0.0;
				ox[inc] = 0.0;
				oy[inc] = 2 * z * sin(y);
				oz[inc] = (1 - z*z)*cos(y);
			}

	cuCheck(cudaMemcpy(pb.dptr_u.ptr, pb.hptr_u, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_v.ptr, pb.hptr_v, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_w.ptr, pb.hptr_w, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_x.ptr, pb.hptr_omega_x, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_y.ptr, pb.hptr_omega_y, size, cudaMemcpyHostToDevice), "memcpy");
	cuCheck(cudaMemcpy(pb.dptr_omega_z.ptr, pb.hptr_omega_z, size, cudaMemcpyHostToDevice), "memcpy");
	
	// transform the initial flow into physical space 
	int dim[3];
	dim[0] = pb.mx;
	dim[1] = pb.my;
	dim[2] = pb.mz;

	int tDim[3];
	tDim[0] = pb.mz;
	tDim[1] = pb.mx;
	tDim[2] = pb.my;

	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);

	//copy initial rhs_v and rhs_omeag_y
	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");

	for (int k = 0; k < pb.nz; k++) {
		pb.tv0[k] = pb.rhs_v[k];
		pb.tomega_y_0[k] = pb.rhs_omega_y[k];
	}

	for (int j = 0; j < pb.my; j++) {
		for (int i = 0; i < (pb.mx / 2 + 1); i++) {
			for (int k = 0; k < pb.mz; k++) {
				size_t inc = pb.tPitch / sizeof(complex)*(j*(pb.mx / 2 + 1) + i) + k;
				pb.rhs_v_p[inc] = pb.rhs_v[inc];
			}
		}
	}

	safeFree(pb.hptr_u);
	safeFree(pb.hptr_v);
	safeFree(pb.hptr_w);
	safeFree(pb.hptr_omega_x);
	safeFree(pb.hptr_omega_y);
	safeFree(pb.hptr_omega_z);
}

void compare_rhs_v(problem& pb) {

}