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
#include "velocity.h"
#include <fstream>
using namespace std;

void write_rhs(problem & pb, string fname, int ix, int iy);
void write_omega(problem& pb, string fname, int ix, int iy);
void write_lamb(problem& pb, string fname, int ix, int iy);
void write_plamb(problem& pb, string fname, int ix, int iy);
void write_tu(problem& pb, string fname, int ix, int iy);
void write_nonlinear(problem& pb, string fname, int ix, int iy);

TestResult test_rhs() {

	RPCF_Paras para;
	para.numPara.mx = 96;
	para.numPara.my = 96;
	para.numPara.mz = 120;
	para.numPara.dt = 0.005;
	//para.numPara.Re = 2600;
	para.numPara.Re = 2600;
	para.numPara.n_pi_x = 8;
	para.numPara.n_pi_y = 4;
	//para.numPara.Ro = 0.01;
	para.numPara.Ro = 0.0;
	problem pb(para);
	
	//RPCF_Paras paras("parameter.txt");
	////paras.numPara.dt = 0.0;
	//problem pb(paras);
	cout << "init cuda" << endl;
	setupCUDA(pb);
	initCUDA(pb);
	cout << "init fft" << endl;
	initFFT(pb);
	cout << "init solver" << endl;
	initSolver(pb);
	cout << "init flow" << endl;
	initFlow(pb);
	//compare_rhs_v(pb);
	/*for (int i = 0; i < 10; i++) {
		std::cout << "step: " << i << std::endl;
		nextStep(pb);
		output_velocity(pb);
	}*/
	nextStep(pb);

	destroyMyCudaMalloc();

	//nextStep(pb);

	initCUDA(pb);

	cout << "init flow" << endl;
	initFlow(pb);
	pb.currenStep = 0;
	cout << "writing initial field to files ..." ;
	//output_velocity(pb);
	cout << "finishied!" << endl;
	//RPCF::write_3d_to_file("rhs_v_pre.txt", (REAL*)pb.rhs_v,
	//	pb.tPitch, 2 * pb.nz, (pb.nx / 2 + 1), pb.ny);

	//nextStep(pb);
	//pb.currenStep++;
	//nextStep(pb);
	//pb.currenStep++;
	//nextStep(pb);
	//pb.currenStep++;

	write_rhs(pb, "rhs_x_1_y_1_LINEAR_PRE.txt", 1, 1);
	
	//write_omega(pb, "omega_x_x_1_y_1.txt", 1, 1);

	get_rhs_v(pb);
	//transform(BACKWARD, pb);
	//computeLambVector(pb);
	//transform(FORWARD, pb); 
	//rhsNonlinear(pb);
	//cheby_s2p(pb.dptr_tLamb_x, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);
	//cheby_s2p(pb.dptr_tLamb_y, pb.mx / 2 + 1, pb.my, pb.mz, No_Padding);
	//write_nonlinear(pb, "rhsNonlinear_x_1_y_1.txt", 1, 1);
	//write_nonlinear(pb, "rhsNonlinear_x_2_y_1.txt", 2, 1);
	//write_nonlinear(pb, "rhsNonlinear_x_1_y_2.txt", 1, 2);
	//write_nonlinear(pb, "rhsNonlinear_x_1_y_3.txt", 1, 3);
	//write_nonlinear(pb, "rhsNonlinear_x_19_y_1.txt", 19, 1);
	//write_nonlinear(pb, "rhsNonlinear_x_20_y_1.txt", 20, 1);

	//write_tu(pb, "tu_x_1_y_1.txt", 1, 1);

	//write_plamb(pb, "lamb_x_1_y_1.txt", 1, 1);
	//write_lamb(pb, "lamb_x_1_y_1.txt", 1, 1);

	write_rhs(pb, "rhs_x_1_y_0_LINEAR.txt", 1, 0);
	write_rhs(pb, "rhs_x_1_y_1_LINEAR.txt", 1, 1);
	//write_rhs(pb, "rhs_x_1_y_1.txt", 1, 1);
	//RPCF::write_3d_to_file("rhs_v.txt", (REAL*)pb.rhs_v,
	//	pb.tPitch, 2 * pb.nz, (pb.nx / 2 + 1), pb.ny);
	//RPCF::write_3d_to_file("nonliear_v.txt", (REAL*)pb.nonlinear_v,
	//	pb.tPitch, 2 * pb.nz, (pb.nx / 2 + 1), pb.ny);


	solveEq(pb.matrix_coeff_v, pb.rhs_v,
		pb.nz, pb.tPitch, pb.mx, pb.my);

	get_rhs_omega(pb);

	solveEq(pb.matrix_coeff_omega, pb.rhs_omega_y, pb.nz, pb.tPitch, pb.mx, pb.my);

	save_0_v_omega_y(pb);

	synchronizeGPUsolver();

	cuCheck(cudaMemcpy(pb.dptr_tw.ptr, pb.rhs_v, pb.tSize, cudaMemcpyHostToDevice), "cpy");
	cuCheck(cudaMemcpy(pb.dptr_tomega_z.ptr, pb.rhs_omega_y, pb.tSize, cudaMemcpyHostToDevice), "cpy");

	getUVW(pb);
	pb.currenStep++;


	//step 2

	get_rhs_v(pb);
	write_rhs(pb, "rhs_x_1_y_0_LINEAR_step2.txt", 1, 0);
	write_rhs(pb, "rhs_x_1_y_1_LINEAR_step2.txt", 1, 1);

	//compare_rhs_v(pb);

	//RPCF::write_3d_to_file("rhs_omega.txt", (REAL*)pb.rhs_omega_y,
	//	pb.tPitch, 2 * pb.nz, (pb.nx / 2 + 1), pb.ny);
	//RPCF::write_3d_to_file("nonliear_omega.txt", (REAL*)pb.nonlinear_omega_y,
	//	pb.tPitch, 2 * pb.nz, (pb.nx / 2 + 1), pb.ny);
	return TestSuccess;
}

void set_flow_rhs_test(problem& pb)
{
	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pz = pb.pz;
	int pitch = pb.pitch;
	REAL lx = pb.lx;
	REAL ly = pb.ly;

	pb.hptr_u = (REAL*)malloc(pb.size);
	pb.hptr_v = (REAL*)malloc(pb.size);
	pb.hptr_w = (REAL*)malloc(pb.size);
	pb.hptr_omega_x = (REAL*)malloc(pb.size);
	pb.hptr_omega_y = (REAL*)malloc(pb.size);
	pb.hptr_omega_z = (REAL*)malloc(pb.size);

	REAL* u = pb.hptr_u;
	REAL* v = pb.hptr_v;
	REAL* w = pb.hptr_w;
	REAL* ox = pb.hptr_omega_x;
	REAL* oy = pb.hptr_omega_y;
	REAL* oz = pb.hptr_omega_z;

	size_t size = pitch * my * mz;

	REAL PI = 4.0*atan(1.0);
	for (int k = 0; k < pz; k++)
		for (int j = 0; j < my; j++)
			for (int i = 0; i < mx; i++)
			{
				REAL x = lx * i / mx;
				REAL y = ly * j / my;
				REAL z = cos(REAL(k) / (pz - 1)*PI);
				size_t inc = (pitch * my * k + pitch *j) / sizeof(REAL) + i;
				//u[inc] = (1 - z*z)*sin(y);
				v[inc] = 0.0;
				u[inc] = 0.0;
				w[inc] = (1 - z*z)*sin(y);
				//w[inc] = 0.0;
				ox[inc] = 0.0;
				oy[inc] = -2 * z * sin(y);
				oz[inc] = -(1 - z*z)*cos(y);
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

	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);


	//copy initial rhs_v and rhs_omeag_y
	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");

	for (int k = 0; k < pb.nz; k++) {
		pb.tv0[k] = pb.rhs_v[k];
		pb.tomega_y_0[k] = pb.rhs_omega_y[k];
	}

	for (int j = 0; j < pb.ny; j++) {
		for (int i = 0; i < (pb.nx / 2 + 1); i++) {
			for (int k = 0; k < pb.nz; k++) {
				size_t inc = pb.tPitch / sizeof(cuRPCF::complex)*(j*(pb.nx / 2 + 1) + i) + k;
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

void write_rhs(problem & pb,string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	cuRPCF::complex* matrix = pb.rhs_v + pb.tPitch/sizeof(cuRPCF::complex)*((pb.nx/2+1)*iy + ix);
	//cuRPCF::complex* matrix = pb.rhs_v;
	cout << "writing: " << fname << endl;
	for (int i = 0; i < pb.pz; i++) {
		outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	}
	outfile.close();
}

void write_omega(problem& pb, string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	cuRPCF::complex* omega_x = (cuRPCF::complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(omega_x, pb.dptr_tomega_x.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuRPCF::complex* matrix = omega_x + pb.tPitch / sizeof(cuRPCF::complex)*((pb.nx / 2 + 1)*iy + ix);
	//cuRPCF::complex* matrix = pb.rhs_v;
	cout << "writing: " << fname << endl;
	for (int i = 0; i < pb.pz; i++) {
		outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	}
	outfile.close();
	free(omega_x);
}

void write_tu(problem& pb, string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	cuRPCF::complex* tu = (cuRPCF::complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(tu, pb.dptr_tv.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuRPCF::complex* matrix = tu + pb.tPitch / sizeof(cuRPCF::complex)*((pb.nx / 2 + 1)*iy + ix);
	//cuRPCF::complex* matrix = pb.rhs_v;
	cout << "writing: " << fname << endl;
	for (int i = 0; i < pb.pz; i++) {
		outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	}
	outfile.close();
	free(tu);
}

void write_lamb(problem& pb, string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	cuRPCF::complex* lamb = (cuRPCF::complex*)malloc(pb.tSize);
	//cuRPCF::complex* lamb = (cuRPCF::complex*)malloc(pb.pSize);
	cuCheck(cudaMemcpy(lamb, pb.dptr_tLamb_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuRPCF::complex* matrix = lamb + pb.tPitch / sizeof(cuRPCF::complex)*((pb.nx / 2 + 1)*iy + ix);
	//cuCheck(cudaMemcpy(lamb, pb.dptr_u.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	//ix = 15;
	//REAL* matrix = ((REAL*)lamb) + (pb.pitch/sizeof(REAL)*(pb.py*ix + iy));
	cout << "writing: " << fname << endl;
	for (int i = 0; i < pb.pz; i++) {
		outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	}
	//for (int i = 0; i < pb.pz; i++) {
	//	outfile << *(((REAL*)lamb) + (pb.pitch / sizeof(REAL)*(pb.py*i + iy)) + 10) << endl;
	//}
	outfile.close();
	free(lamb);
}

void write_plamb(problem& pb, string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	//cuRPCF::complex* lamb = (cuRPCF::complex*)malloc(pb.tSize);
	cuRPCF::complex* lamb = (cuRPCF::complex*)malloc(pb.pSize);
	//cuCheck(cudaMemcpy(lamb, pb.dptr_tLamb_z.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	//cuRPCF::complex* matrix = lamb + pb.tPitch / sizeof(cuRPCF::complex)*((pb.nx / 2 + 1)*iy + ix);
	cuCheck(cudaMemcpy(lamb, pb.dptr_lamb_z.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	//ix = 15;
	REAL* matrix = ((REAL*)lamb) + (pb.pitch/sizeof(REAL)*(pb.py*ix + iy));
	cout << "writing: " << fname << endl;
	//for (int i = 0; i < pb.pz; i++) {
	//	outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	//}
	for (int i = 0; i < pb.pz; i++) {
		outfile << *(((REAL*)lamb) + (pb.pitch / sizeof(REAL)*(pb.py*i + iy)) + 1) << endl;
	}
	outfile.close();
	free(lamb);
}

void write_nonlinear(problem& pb, string fname, int ix, int iy) {
	ofstream outfile(fname.c_str(), fstream::out);
	assert(outfile.is_open());
	cuRPCF::complex* nonlinear = (cuRPCF::complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(nonlinear, pb.dptr_tLamb_x.ptr, pb.tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuRPCF::complex* matrix = nonlinear + pb.tPitch / sizeof(cuRPCF::complex)*((pb.nx / 2 + 1)*iy + ix);
	//cuRPCF::complex* matrix = pb.rhs_v;
	cout << "writing: " << fname << endl;
	for (int i = 0; i < pb.pz; i++) {
		outfile << matrix[i].re << "\t" << matrix[i].im << endl;
	}
	outfile.close();
	free(nonlinear);
}