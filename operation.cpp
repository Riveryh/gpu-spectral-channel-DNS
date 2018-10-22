#include "operation.h"
#include "data.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
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
	//return 0;
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


void cuCheck(cudaError_t ret, string s) {
	if (ret == cudaSuccess) {
		return;
	}
	else {
		printf("cudaError at %s\n", s.c_str());
		assert(false);
	}
}

bool isEqual(real a, real b, real precision ){
	if (abs(a - b) <= precision) {
		return true;
	}
	else
	{
		if (abs(a/b-1.0)<1e-4) {
			return true;
		}
		else {
			return false;
		}
	}
}



void RPCF_Paras::read_para(std::string filename) {
	ifstream infile;
	infile.open(filename, ios::in);
	if (!infile.is_open()) {
		cerr << "Error in opening file" << endl;
		//exit(-1);
		return;
	}
	RPCF_Numerical_Para& np = this->numPara;
	
	infile >> np.mx >> np.my >> np.mz;
	infile >> np.n_pi_x >> np.n_pi_y;
	infile >> np.Re >> np.Ro >> np.dt;

	RPCF_Step_Para& sp = this->stepPara;
	infile >> sp.start_type >> sp.start_step 
		>> sp.end_step >> sp.save_internal >> sp.save_recovery_internal;

	RPCF_IO_Para& iop = this->ioPara;
	infile >> iop.output_file_prefix;
	infile >> iop.recovery_file_prefix;

	infile.close();
}


cudaPitchedPtr __myPtr;
size_t __myPPitch;
size_t __myTPitch;
int __my_pMem_allocated;
int __my_tMem_allocated;
int __my_pSize;
int __my_tSize;
size_t __myMaxMemorySize;

__host__ void initMyCudaMalloc(dim3 dims) {
	// get memory allignment factor
	cudaPitchedPtr testPtr;
	
	int mx = dims.x;
	int my = dims.y;
	int mz = dims.z;
	int nx = mx/ 3 * 2;
	int ny = my/ 3 * 2;
	int pz = mz / 2 + 1;

	cudaExtent ext = make_cudaExtent(
		sizeof(complex)*(mx/2+1),my,pz
	);
	cuCheck(cudaMalloc3D(&testPtr, ext), "mem test");
	__myPPitch = testPtr.pitch;
	cudaFree(testPtr.ptr);

	ext = make_cudaExtent(
		sizeof(complex)*mz, nx / 2 + 1, ny
	);
	cuCheck(cudaMalloc3D(&testPtr, ext), "mem test");
	__myTPitch = testPtr.pitch;
	cudaFree(testPtr.ptr);

	__my_pSize = __myPPitch * my * pz;
	__my_tSize = __myTPitch * (nx / 2 + 1) * ny;
	size_t maxSize = __my_pSize>__my_tSize ? __my_pSize : __my_tSize;

	__myMaxMemorySize = maxSize * 8;

	// mallocate the whole memory at one time to save time.
	ext = make_cudaExtent(__myMaxMemorySize, 1, 1);
	cuCheck(cudaMalloc3D(&__myPtr, ext),"my cuda malloc");

	cuCheck(cudaMemset(__myPtr.ptr, -1, __myMaxMemorySize),"memset");

	__my_pMem_allocated = 0;
	__my_tMem_allocated = 0;
}

__host__ void* get_fft_buffer_ptr() {
	return __myPtr.ptr;
}

__host__ cudaError_t myCudaMalloc(cudaPitchedPtr& Ptr, myCudaMemType type) {
	if (__my_pMem_allocated + __my_tMem_allocated >= 7)return cudaErrorMemoryAllocation;
	if (type == XYZ_3D) {
		// check if memory is already used up.
		if (__my_pMem_allocated >= 6)return cudaErrorMemoryAllocation;
		Ptr.pitch = __myPPitch;
		Ptr.ptr = (char*)__myPtr.ptr + (__my_pMem_allocated+1)*__my_pSize;
		__my_pMem_allocated++;
		return cudaSuccess;
	}
	else if (type == ZXY_3D) {
		// check if memory is already used up.
		if (__my_tMem_allocated >= 6)return cudaErrorMemoryAllocation;
		Ptr.pitch = __myTPitch;
		Ptr.ptr = (char*)__myPtr.ptr + __myMaxMemorySize
			- (__my_tMem_allocated+1)*__my_tSize;
		__my_tMem_allocated++;
		return cudaSuccess;
	}
	else {
		return cudaErrorInvalidValue;	//WRONG TYPE	
	}
}

__host__ cudaError_t myCudaFree(cudaPitchedPtr& Ptr, myCudaMemType type) {
	if (type == XYZ_3D) {
		if (__my_pMem_allocated <= 0) return cudaErrorMemoryAllocation;
		int i = ((char*)Ptr.ptr - (char*)__myPtr.ptr) / __my_pSize;
		assert(((char*)Ptr.ptr - (char*)__myPtr.ptr) % __my_pSize == 0);
		// the next memory to free must be the last memory of allocated block.
		if (__my_pMem_allocated != i) return cudaErrorInvalidValue;
		Ptr.ptr = NULL;
		__my_pMem_allocated--;
		return cudaSuccess;
	}
	else if (type == ZXY_3D) {
		if (__my_tMem_allocated <= 0) return cudaErrorMemoryAllocation;
		int i = ((char*)__myPtr.ptr + __myMaxMemorySize - (char*)Ptr.ptr) / __my_tSize;
		assert(((char*)__myPtr.ptr + __myMaxMemorySize - (char*)Ptr.ptr) % __my_tSize == 0);
		if (__my_tMem_allocated != i) return cudaErrorInvalidValue;
		Ptr.ptr = NULL;
		__my_tMem_allocated--;
		return cudaSuccess;
	}
	else {
		return cudaErrorInvalidValue;
	}
}

__host__ void destroyMyCudaMalloc() {
	cuCheck(cudaFree(__myPtr.ptr),"destroy allocator");
}

string get_file_name(string prefix, int num, string suffix) {
	string filename;
	ostringstream s_num;
	s_num << num;
	filename = prefix + s_num.str() + string(".") + suffix;
	return filename;
}