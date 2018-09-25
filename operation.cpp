#include "operation.h"
#include "data.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include "parameters.h"

using namespace std;

//int RPCF::read_parameter(Flow& fl,std::string& s) {
//	parameter& para = fl.para;
//	para.dt = 0.005;
//	para.Re = 1300;
//	para.Ro = 0.01;
//	fl.nnx = 256;
//	fl.nny = 256;
//	fl.nnz = 70;
//	return 0;
//}
//
//Flow::Flow(int i, int j, int k) :
//	u(i,j,k),v(i,j,k),w(i,j,k),
//	omega_x(i,j,k),omega_y(i,j,k),omega_z(i,j,k),
//	u_k(i,j,k),v_k(i,j,k),w_k(i,j,k),
//	omega_x_k(i,j,k),omega_y_k(i, j, k), omega_z_k(i, j, k)
//{
//};
//
//int Flow::initialize() {
//	return 0;
//}

void problem::memcpy_device_to_host() {
	size_t isize = dptr_u.pitch*ny*nz;
	cudaError_t cuerr;
	//isize = 1 * sizeof(real);
	if (hptr_omega_z==nullptr) {		
		std::cout << "allocated " << endl;
		hptr_u = (real*)malloc(isize);
		hptr_v = (real*)malloc(isize);
		hptr_w = (real*)malloc(isize);
		hptr_omega_x = (real*)malloc(isize);
		hptr_omega_y = (real*)malloc(isize);
		hptr_omega_z = (real*)malloc(isize);
	}
	cuerr = cudaSuccess;
	cuerr = cudaMemcpy(hptr_u, dptr_u.ptr, isize, cudaMemcpyDeviceToHost);
	cuerr = cudaMemcpy(hptr_v, dptr_v.ptr, isize, cudaMemcpyDeviceToHost);
	cuerr = cudaMemcpy(hptr_w, dptr_w.ptr, isize, cudaMemcpyDeviceToHost);

	cuerr = cudaMemcpy(hptr_omega_x, dptr_omega_x.ptr, isize, cudaMemcpyDeviceToHost);
	cuerr = cudaMemcpy(hptr_omega_y, dptr_omega_y.ptr, isize, cudaMemcpyDeviceToHost);
	//cuerr = cudaMemcpy(hptr_omega_z, dptr_omega_z.ptr, isize, cudaMemcpyDeviceToHost);

	if (cuerr != cudaSuccess) {
		cout << cuerr << endl;
	}
	cudaDeviceSynchronize();
}


int RPCF::write_3d_to_file(char* filename,real* pu, int pitch, int nx, int ny, int nz) {
	ofstream outfile(filename,fstream::out);
	// skip this part
	return 0;
	ASSERT(outfile.is_open());
	for (int k = 0; k < nz; k++) {
		size_t slice = pitch*ny*k;
		for (int j = 0; j < ny; j++) {
			real* row = (real*)((char*)pu + slice + j*pitch);
			for (int i = 0; i < nx; i++) {
				outfile << row[i] << "\t";
			}
			outfile << endl;
		}
		outfile << endl;
	}
	outfile.close();
	return 0;
}


void cuCheck(cudaError_t ret, char* s) {
	if (ret == cudaSuccess) {
		return;
	}
	else {
		printf("cudaError at %s\n", s);
		assert(false);
	}
}

bool isEqual(real a, real b, real precision ){
	if (abs(a - b) <= precision) {
		return true;
	}
	else
	{
		if (abs(abs(a/b)-1.0)<1e-4) {
			return true;
		}
		else {
			return false;
		}
	}
}



void RPCF_Paras::read_para(char* filename) {
	ifstream infile;
	infile.open(filename, ios::in);
	if (!infile.is_open()) {
		cerr << "Error in opening file" << endl;
		exit(-1);
	}
	RPCF_Numerical_Para& np = this->numPara;
	
	infile >> np.nx >> np.ny >> np.nz;
	infile >> np.n_pi_x >> np.n_pi_y;
	infile >> np.Re >> np.Ro >> np.dt;

	RPCF_Step_Para& sp = this->stepPara;
	infile >> sp.start_step >> sp.end_step >> sp.save_internal;

	RPCF_IO_Para& iop = this->ioPara;
	infile >> iop.input_file;
	infile >> iop.output_file;

	infile.close();
}