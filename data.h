#pragma once

#define DEBUG
//#undef DEBUG

#define REAL_DOUBLE
#undef REAL_FLOAT

#ifdef REAL_DOUBLE
#define real double
#undef REAL_FLOAT
#endif
#ifdef REAL_FLOAT
#define real float
#undef REAL_DOUBLE
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "parameters.h"
#include <cmath>

#ifdef DEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#define MAX_NZ 100

#define safeCudaFree(p); {cuCheck(cudaFree(p),"deallocate");(p)=nullptr;}
#define safeFree(p); {free(p);(p)=nullptr;}

struct complex {
	real re;
	real im;
	__host__ __device__ complex() {};
	__host__ __device__ complex(real ire, real iim) :re(ire), im(iim) {};
	__host__ __device__ complex& operator=(complex c) {
		this->re = c.re;
		this->im = c.im;
		return *this;
	}
	__host__ __device__ complex& operator=(real r) {
		this->re = r;
		this->im = 0.0;
		return *this;
	}
	__host__ __device__ complex operator*(real r) {
		real tre = this->re * r;
		real tim = this->im * r;
		return complex(tre, tim);
	}
	__host__ __device__ complex operator*(complex c) {
		real tre = this->re*c.re - this->im*c.im;
		real tim = this->re*c.im + this->im*c.re;
		return complex(tre, tim);
	}
	__host__ __device__ complex operator+(complex c) {
		real tre = this->re + c.re;
		real tim = this->im + c.im;
		return complex(tre, tim);
	}
	__host__ __device__ complex operator-(complex c) {
		real tre = this->re - c.re;
		real tim = this->im - c.im;
		return complex(tre, tim);
	}
	__host__ __device__ complex operator+(real r) {
		real tre = this->re + r;
		real tim = this->im;
		return complex(tre, tim);
	}

};

//
//struct parameter {
//	real Ro;
//	real Re;
//	real dt;
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
	bool _isPrivate = false;
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
//	matrix3d<real> u;
//	matrix3d<real> v;
//	matrix3d<real> w;
//	matrix3d<real> omega_x;
//	matrix3d<real> omega_y;
//	matrix3d<real> omega_z;
//
//	matrix3d<complex> u_k;
//	matrix3d<complex> v_k;
//	matrix3d<complex> w_k;
//	matrix3d<complex> omega_x_k;
//	matrix3d<complex> omega_y_k;
//	matrix3d<complex> omega_z_k;
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
//	int phy_to_spec(real*, complex*);
//	int spec_to_phy();
//	int spec_to_phy(complex*, real*);
//
//	real& get_real(real* ,int i, int j, int k);
//	complex& get_complex(complex*, int i, int j, int k);
//};

struct problem {
	int nx, ny, nz;	// the number of mesh
	int mx, my, mz;	// number of allocated memory
	int px, py, pz; // number of mesh for dealiasing
	const real PI = 4*atan(1.0l);
	real aphi;
	real beta;
	real lx;
	real ly;
	real dt;
	real Re;
	real Ro;
	real *_U0, *_dU0, *_ddU0;
	//
	//matrix2d<real> T0;
	//matrix2d<real> T2;
	//matrix2d<real> T4;
	
	real* T0, *T2, *T4;

	cudaExtent extent;
	cudaExtent tExtent;

	size_t size;	// pb.pitch * pb.my * pb.mz
	size_t tSize;	// pb.tPitch * (pb.mx/2+1) * pb.my

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

	real* dp_meanU;

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
	real* hptr_u;
	real* hptr_v;
	real* hptr_w;
	real* hptr_omega_x;
	real* hptr_omega_y;
	real* hptr_omega_z;

	// host pointer of nonlinear term in v equation
	complex* nonlinear_v;

	// host pointer of nonlinear term in omegaY equation
	complex* nonlinear_omega_y;

	// host pointer of nonlinear term in v equation of previous step
	complex* nonlinear_v_p;

	// host pointer of nonlinear term in omegaY equation of previous step
	complex* nonlinear_omega_y_p;

	complex* matrix_coeff_v;
	complex* matrix_coeff_omega;

	// after solving the equation, the pointer contains v and omega;
	// after running getRHS, this pointer contains the right hand side
	// part value of the equation.
	complex* rhs_v;
	complex* rhs_omega_y;
	complex* rhs_v_p;
	//complex* rhs_omega_y_p;

	// save the zero-wave number lamb vector to solve (0,0) 
	// wave number equation.
	complex* lambx0;
	complex* lambz0;
	complex* lambx0_p;
	complex* lambz0_p;

	//zero-wave nubmer velocity in spectral space
	complex* tv0;
	complex* tomega_y_0;
	
	problem() :
		nx(128),
		ny(128),
		nz(30),
		aphi(4.0),
		beta(2.0)//,
		//T0(nz,nz),
		//T2(nz, nz),
		//T4(nz, nz)
	{
		initVars();
	};

	problem(RPCF_Paras conf) {
		nx = conf.numPara.nx;
		ny = conf.numPara.ny;
		nz = conf.numPara.nz;
		aphi = conf.numPara.n_pi_x / 2;
		beta = conf.numPara.n_pi_y / 2;
		Ro = conf.numPara.Ro;
		dt = conf.numPara.dt;
		Re = conf.numPara.Re;
		initVars();
	}

	void memcpy_device_to_host();

private:
	void initVars() {
		lx = 2 * PI*aphi;
		ly = 2 * PI*beta;
		hptr_omega_z = nullptr;
		mx = nx * 3 / 2;
		my = ny * 3 / 2;
		mz = (nz - 1) * 4;
		px = nx * 3 / 2;
		py = ny * 3 / 2;
		pz = (nz - 1) * 2 + 1;


		// compute cuda dimensions.
		nThread.x = 16;
		nThread.y = 16;
		nDim.x = my / nThread.x;
		nDim.y = mz / nThread.y;
		if (my%nThread.x != 0) nDim.x++;
		if (mz%nThread.y != 0) nDim.y++;

		ntDim.x = (mx/2+1) / nThread.x;
		ntDim.y = my / nThread.y;
		if ((mx / 2 + 1) % nThread.x != 0) ntDim.x++;
		if (my % nThread.y != 0) ntDim.y++;

		npDim.x = py / nThread.x;
		npDim.y = pz / nThread.y;
		if (py%nThread.x != 0) npDim.x++;
		if (pz%nThread.y != 0) npDim.y++;
	}
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

template<class T>
inline void swap(T& a, T& b) {
	T temp;
	temp = a;
	a = b;
	b = temp;
}


#define RPCF_EQUAL_PRECISION 1e-8
bool isEqual(real a, real b, real precision = RPCF_EQUAL_PRECISION);



