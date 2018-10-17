#include "test_coef.h"
#include "coefficient.cuh"
#include <cassert>
#include <fstream>
#include "solver.h"
using namespace std;

TestResult test_get_T_matrix()
{
	real *T0, *T2, *T4;
	int N = 100;
	T0 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	T2 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	T4 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	get_T_matrix(N, T0, T2, T4);
	size_t inc1 = (N + 1) * 16 + 100;
	size_t inc2 = (N + 1) * 42 + 95;
	size_t inc3 = (N + 1) * 0 + 4;
	size_t inc4 = (N + 1) * 100 + 5;
	real PRECISION = 1e-6;
	assert(isEqual(T4[inc1] / 1e9, 1.846564249235145, PRECISION));
	assert(isEqual(T4[inc2] / 1e7, 8.842927487588026, PRECISION));
	assert(isEqual(T4[inc3], 192.0, PRECISION));
	assert(isEqual(T4[inc4], -1920.0, PRECISION));
	for (int i = 0; i <= N; i++) {
		for (int j = 0; j <= 3; j++) {
			size_t inc_j = i*(N + 1) + j;
			assert(isEqual(T4[inc_j],0.0,PRECISION));
		}
	}
	return TestSuccess;
}

TestResult Test_coef_v() {
	RPCF_Paras para;
	para.numPara.mx = 64;
	para.numPara.my = 64;
	para.numPara.mz = 32;
	para.numPara.dt = 0.005;
	para.numPara.Re = 2600;
	para.numPara.n_pi_x = 8;
	para.numPara.n_pi_y = 4;
	para.numPara.Ro = 0.01;
	problem pb(para);

	initSolver(pb, false);
	size_t inc = pb.nz*pb.nz;
	matrix2d<complex> coef_v((pb.matrix_coeff_v + inc), pb.nz, pb.nz);
	int nz = pb.nz;
	char* str;
	str = "test_data\\coef_v\\coef_x_001_y_000.dat";
	ifstream infile;
	infile.open(str);
	if (infile.is_open()) {
		real data;
		for (int i = 0; i < nz; i++) {
			for (int j = 0; j < nz; j++) {
				infile >> data;
				complex res = coef_v(i, j);
				assert(isEqual(data, res.re, 1e-6));
				infile >> data;
				assert(isEqual(data, res.im, 1e-6));
			}
			//cout << "row:" << i << " cleared" << endl;
		}
	}
	else
	{
		cerr << "cannot open coef file" << endl;
		return TestFailed;
	}

	// test matrix 2;
	str = "test_data\\coef_v\\coef_x_017_y_063.dat";
	inc = pb.nz*pb.nz*((pb.mx / 2 + 1)*(63 + 32) + 17);
	matrix2d<complex> coef_v2((pb.matrix_coeff_v + inc), pb.nz, pb.nz);
	infile.close();
	infile.open(str);
	if (infile.is_open()) {
		real data;
		for (int i = 0; i < nz; i++) {
			for (int j = 0; j < nz; j++) {
				infile >> data;
				complex res = coef_v2(i, j);
				assert(isEqual(data, res.re, 1e-6));
				infile >> data;
				assert(isEqual(data, res.im, 1e-6));
			}
			//cout << "row:" << i << " cleared" << endl;
		}
	}
	else
	{
		cerr << "cannot open coef file" << endl;
		return TestFailed;
	}
	return TestSuccess;
}

TestResult Test_coef_omega() {
	RPCF_Paras para;
	para.numPara.mx = 64;
	para.numPara.my = 64;
	para.numPara.mz = 51;
	para.numPara.dt = 0.005;
	para.numPara.Re = 2600;
	para.numPara.n_pi_x = 8;
	para.numPara.n_pi_y = 4;
	para.numPara.Ro = 0.01;
	problem pb(para);

	initSolver(pb, false);
	size_t inc = pb.nz*pb.nz;
	matrix2d<complex> coef_omega((pb.matrix_coeff_omega + inc), pb.nz, pb.nz);
	int nz = pb.nz;
	char* str;
	str = "test_data\\coef_omega\\coef_x_001_y_000.dat";
	ifstream infile;
	infile.open(str);
	if (infile.is_open()) {
		real data;
		for (int i = 0; i < nz; i++) {
			for (int j = 0; j < nz; j++) {
				infile >> data;
				complex res = coef_omega(i, j);
				assert(isEqual(data, res.re, 1e-6));
				infile >> data;
				assert(isEqual(data, res.im, 1e-6));
			}
			//cout << "row:" << i << " cleared" << endl;
		}
	}
	else
	{
		cerr << "cannot open coef file" << endl;
		return TestFailed;
	}

	// test matrix 2;
	str = "test_data\\coef_omega\\coef_x_017_y_063.dat";
	inc = pb.nz*pb.nz*((pb.mx / 2 + 1)*(63 + 32) + 17);
	matrix2d<complex> coef_omega2((pb.matrix_coeff_omega + inc), pb.nz, pb.nz);
	infile.close();
	infile.open(str);
	if (infile.is_open()) {
		real data;
		for (int i = 0; i < nz; i++) {
			for (int j = 0; j < nz; j++) {
				infile >> data;
				complex res = coef_omega2(i, j);
				assert(isEqual(data, res.re, 1e-6));
				infile >> data;
				assert(isEqual(data, res.im, 1e-6));
			}
			//cout << "row:" << i << " cleared" << endl;
		}
	}
	else
	{
		cerr << "cannot open coef file" << endl;
		return TestFailed;
	}
	return TestSuccess;
}

