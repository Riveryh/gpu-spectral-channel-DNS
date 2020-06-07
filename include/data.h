#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

#include "parameters.h"

#define REAL_DOUBLE
//#define REAL_FLOAT
#undef REAL_FLOAT

#ifdef REAL_DOUBLE
#define REAL double
#undef REAL_FLOAT
#endif
#ifdef REAL_FLOAT
#define REAL float
#undef REAL_DOUBLE
#endif

#ifndef NDEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#define MAX_NZ 100

#define safeCudaFree(p); {cuCheck(cudaFree(p),"deallocate");(p)=nullptr;}
#define safeFree(p); {free(p);(p)=nullptr;}

#define nullptr NULL

namespace cuRPCF{
	struct complex {
		REAL re;
		REAL im;
		__host__ __device__ complex() {};
		__host__ __device__ complex(REAL ire, REAL iim) :re(ire), im(iim) {};
		__host__ __device__ complex& operator=(complex c) {
			this->re = c.re;
			this->im = c.im;
			return *this;
		}
		__host__ __device__ complex& operator=(REAL r) {
			this->re = r;
			this->im = 0.0;
			return *this;
		}
		__host__ __device__ complex operator*(REAL r) {
			REAL tre = this->re * r;
			REAL tim = this->im * r;
			return cuRPCF::complex(tre, tim);
		}
		__host__ __device__ complex operator*(complex c) {
			REAL tre = this->re * c.re - this->im * c.im;
			REAL tim = this->re * c.im + this->im * c.re;
			return cuRPCF::complex(tre, tim);
		}
		__host__ __device__ complex operator+(complex c) {
			REAL tre = this->re + c.re;
			REAL tim = this->im + c.im;
			return cuRPCF::complex(tre, tim);
		}
		__host__ __device__ complex operator-(complex c) {
			REAL tre = this->re - c.re;
			REAL tim = this->im - c.im;
			return cuRPCF::complex(tre, tim);
		}
		__host__ __device__ complex operator+(REAL r) {
			REAL tre = this->re + r;
			REAL tim = this->im;
			return cuRPCF::complex(tre, tim);
		}
	};
}

//
//struct parameter {
//	REAL Ro;
//	REAL Re;
//	REAL dt;
//	int start_step;
//	int save_step;
//	int output_step;
//	int end_step;
//};

template<class T>
class matrix3d {
	T* mat;
public:
	int nx, ny, nz;
	matrix3d<T>(int i, int j, int k) : nx(i), ny(j), nz(k) {
		size_t tsize = sizeof(T)*nx*ny*nz;
		mat = (T*)malloc(tsize);
	};
	T& get(int i, int j, int k) {

		return *(mat + i*(ny*nz) + j*nz + k);
	}
	T& operator()(int i, int j, int k) {
		return get(i, j, k);
	}
	matrix3d<T>& operator=(T value) {
		for (int i = 0; i < nx; i++)
			for (int j = 0; j < ny; j++)
				for (int k = 0; k < nz; k++) {
					(*this)(i, j, k) = value;
				}
		return *this;
	}
};

template<class T>
class matrix2d {
	T* mat;
	bool _isPrivate;
public: 
	int nx, ny;
	matrix2d<T>(int i, int j) : nx(i), ny(j) {
		size_t tsize = sizeof(T)*nx*ny;
		mat = (T*)malloc(tsize);
		_isPrivate = true;
	};
	matrix2d<T>(T* _mat,int nx, int ny) {
		this->mat = _mat;
		this->nx = nx;
		this->ny = ny;
		_isPrivate = false;
	}
	T& get(int i, int j) {
		return *(mat + i*(ny) + j );
	}
	T& operator()(int i, int j) {
		return get(i, j);
	}
	matrix2d<T>& operator=(T value) {
		for (int i = 0; i < nx; i++)
			for (int j = 0; j < ny; j++) {
					(*this)(i, j) = value;
				}
		return *this;
	}
	~matrix2d() {
		if (_isPrivate) {
			free(mat);
		}
	}
};

//struct Flow {
//	parameter para;
//	int		nnx, nny, nnz;
//	matrix3d<REAL> u;
//	matrix3d<REAL> v;
//	matrix3d<REAL> w;
//	matrix3d<REAL> omega_x;
//	matrix3d<REAL> omega_y;
//	matrix3d<REAL> omega_z;
//
//	matrix3d<cuRPCF::complex> u_k;
//	matrix3d<cuRPCF::complex> v_k;
//	matrix3d<cuRPCF::complex> w_k;
//	matrix3d<cuRPCF::complex> omega_x_k;
//	matrix3d<cuRPCF::complex> omega_y_k;
//	matrix3d<cuRPCF::complex> omega_z_k;
//
//	Flow(int i, int j, int k);
//
//	int initialize();
//
//	int computeRHS();
//
//	int solveEQ();
//
//	int phy_to_spec();
//	int phy_to_spec(REAL*, cuRPCF::complex*);
//	int spec_to_phy();
//	int spec_to_phy(cuRPCF::complex*, REAL*);
//
//	REAL& get_real(REAL* ,int i, int j, int k);
//	cuRPCF::complex& get_complex(cuRPCF::complex*, int i, int j, int k);
//};

struct problem {
	int nx, ny, nz;	// the number of mesh
	int mx, my, mz;	// number of allocated memory
	int px, py, pz; // number of mesh for dealiasing
	//const REAL PI = 4*atan(1.0l);
	REAL aphi;
	REAL beta;
	REAL lx;
	REAL ly;
	REAL dt;
	REAL Re;
	REAL Ro;
	REAL *_U0, *_dU0, *_ddU0;
	int currenStep;
	RPCF_Paras para;
	//
	//matrix2d<REAL> T0;
	//matrix2d<REAL> T2;
	//matrix2d<REAL> T4;
	
	REAL* T0, *T2, *T4;

	cudaExtent extent;
	cudaExtent tExtent;
	cudaExtent pExtent;

	size_t size;	// pb.pitch * pb.my * pb.mz
	size_t tSize;	// pb.tPitch * (pb.mx/2+1) * pb.my
	size_t pSize;   // pb.pitch * pb.my * pb.pz

	size_t pitch, tPitch;

	dim3 nThread;

	dim3 nDim;  // mx my mz
	dim3 ntDim; // mz mx/2+1 my
	dim3 npDim;	// px py pz

	// device pointer of u,v,w,...
	cudaPitchedPtr dptr_u;
	cudaPitchedPtr dptr_v;
	cudaPitchedPtr dptr_w;
	cudaPitchedPtr dptr_omega_x;
	cudaPitchedPtr dptr_omega_y;
	cudaPitchedPtr dptr_omega_z;
	cudaPitchedPtr dptr_lamb_x;
	cudaPitchedPtr dptr_lamb_y;
	cudaPitchedPtr dptr_lamb_z;

	REAL* dp_meanU;

	// device pointer of transposed u,v,w,...
	cudaPitchedPtr dptr_tu;
	cudaPitchedPtr dptr_tv;
	cudaPitchedPtr dptr_tw;
	cudaPitchedPtr dptr_tomega_x;
	cudaPitchedPtr dptr_tomega_y;
	cudaPitchedPtr dptr_tomega_z;
	cudaPitchedPtr dptr_tLamb_x;
	cudaPitchedPtr dptr_tLamb_y;
	cudaPitchedPtr dptr_tLamb_z;

	// device pointer of rhs terms
	cudaPitchedPtr dptr_rhs_v;
	cudaPitchedPtr dptr_rhs_omega_y;
	//cudaPitchedPtr dptr_rhs_v_p;
	//cudaPitchedPtr dptr_rhs_omega_y_p;

	// host pointer of u,v,w,... used for stroage of output.
	REAL* hptr_u;
	REAL* hptr_v;
	REAL* hptr_w;
	REAL* hptr_omega_x;
	REAL* hptr_omega_y;
	REAL* hptr_omega_z;

	// host pointer of nonlinear term in v equation
	cuRPCF::complex* nonlinear_v;

	// host pointer of nonlinear term in omegaY equation
	cuRPCF::complex* nonlinear_omega_y;

	// host pointer of nonlinear term in v equation of previous step
	cuRPCF::complex* nonlinear_v_p;

	// host pointer of nonlinear term in omegaY equation of previous step
	cuRPCF::complex* nonlinear_omega_y_p;

	cuRPCF::complex* matrix_coeff_v;
	cuRPCF::complex* matrix_coeff_omega;

	// after solving the equation, the pointer contains v and omega;
	// after running getRHS, this pointer contains the right hand side
	// part value of the equation.
	cuRPCF::complex* rhs_v;
	cuRPCF::complex* rhs_omega_y;
	cuRPCF::complex* rhs_v_p;
	//cuRPCF::complex* rhs_omega_y_p;

	// save the zero-wave number lamb vector to solve (0,0) 
	// wave number equation.
	cuRPCF::complex* lambx0;
	cuRPCF::complex* lambz0;
	cuRPCF::complex* lambx0_p;
	cuRPCF::complex* lambz0_p;

	//zero-wave nubmer velocity in spectral space
	cuRPCF::complex* tv0;
	cuRPCF::complex* tomega_y_0;
	
	problem() :
		mx(128),
		my(128),
		mz(30),
		aphi(4.0),
		beta(2.0)//,
		//T0(nz,nz),
		//T2(nz, nz),
		//T4(nz, nz)
	{
		initVars();
	};

	problem(RPCF_Paras conf) {
		mx = conf.numPara.mx;
		my = conf.numPara.my;
		mz = conf.numPara.mz;

		nx = conf.numPara.mx * 2/3;
		ny = conf.numPara.my * 2/3;
		nz = conf.numPara.mz / 4 + 1;
		
		aphi = conf.numPara.n_pi_x / 2;
		beta = conf.numPara.n_pi_y / 2;
		Ro = conf.numPara.Ro;
		dt = conf.numPara.dt;
		Re = conf.numPara.Re;
		para = conf;
		initVars();
	}

	//void memcpy_device_to_host();

private:
	void initVars();
};


#define TRANSPOSED true

struct cudaPitchedPtrList {
	cudaPitchedPtr dptr_u;
	cudaPitchedPtr dptr_v;
	cudaPitchedPtr dptr_w;
	cudaPitchedPtr dptr_omega_x;
	cudaPitchedPtr dptr_omega_y;
	cudaPitchedPtr dptr_omega_z;
	cudaPitchedPtr dptr_lamb_x;
	cudaPitchedPtr dptr_lamb_y;
	cudaPitchedPtr dptr_lamb_z;
	cudaPitchedPtrList() {};
	cudaPitchedPtrList(problem& pb,bool isTranposed) {
		if (isTranposed) {
			dptr_u = pb.dptr_tu;
			dptr_v = pb.dptr_tv;
			dptr_w = pb.dptr_tw;
			dptr_omega_x = pb.dptr_tomega_x;
			dptr_omega_y = pb.dptr_tomega_y;
			dptr_omega_z = pb.dptr_tomega_z;
			dptr_lamb_x = pb.dptr_tLamb_x;
			dptr_lamb_y = pb.dptr_tLamb_y;
			dptr_lamb_z = pb.dptr_tLamb_z;
		}
		else {
			dptr_u = pb.dptr_u;
			dptr_v = pb.dptr_v;
			dptr_w = pb.dptr_w;
			dptr_omega_x = pb.dptr_omega_x;
			dptr_omega_y = pb.dptr_omega_y;
			dptr_omega_z = pb.dptr_omega_z;
			dptr_lamb_x = pb.dptr_lamb_x;
			dptr_lamb_y = pb.dptr_lamb_y;
			dptr_lamb_z = pb.dptr_lamb_z;
		}
	};
};

//
//template<class T>
//inline void setMatrix(T* p, int nx, int ny, int nz,
//T *func(int, int, int)) {
//	for (int k = 0; k < nz; k++) {
//		for (int j = 0; j < ny; j++) {
//			for (int i = 0; i < nx; i++) {
//				p[k*nx*ny + j*nx + i] = func(nx, ny, nz);
//			}
//		}
//	}
//}
//
//template<class T>
//inline void setPitchedMatrix(T* p,int pitch, int nx, int ny, int nz,
//	T func(int, int, int,int,int,int)) {
//	int snx = pitch / sizeof(T);
//	for (int k = 0; k < nz; k++) {
//		for (int j = 0; j < ny; j++) {
//			for (int i = 0; i < nx; i++) {
//				p[k*snx*ny + j*snx + i] 
//					= func(nx, ny, nz);
//			}
//		}
//	}
//}

namespace cuRPCF{
	template<class T>
	inline void swap(T& a, T& b) {
		T temp;
		temp = a;
		a = b;
		b = temp;
	}
}

#define RPCF_EQUAL_PRECISION 1e-8
bool isEqual(REAL a, REAL b, REAL precision = RPCF_EQUAL_PRECISION);



