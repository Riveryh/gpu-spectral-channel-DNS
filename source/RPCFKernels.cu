#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/data.h"
#include "../include/RPCFKernels.cuh"
#include "../include/transform.cuh"
#include "../include/operation.h"
#include "../include/cuRPCF.h"
#include <cstdlib>
#include <math.h>
#include <cassert>
#include <iostream>
#include "../include/rhs.cuh"
#include "../include/transform_multi_gpu.h"
#include "../include/velocity.h"


__host__ void getDeviceInfo(problem& pb) {
	cudaDeviceProp prop;
	int dev_num;
	int n_dev;
	size_t free;
	size_t total;
	init_pthread(pb);
	cudaGetDevice(&dev_num);
	cudaGetDeviceProperties(&prop, dev_num);
	cudaMemGetInfo(&free, &total);
	cudaGetDeviceCount(&n_dev);
	//err = cudaDeviceReset();
	//ASSERT(err == cudaSuccess);
	printf("Using CUDA device %u. Device ID: %s on PCI-E %d\n",
		dev_num, prop.name, prop.pciBusID);
	printf("GPU total memory = % .2f MB\n", (float)total / (1.024e6));
	printf("GPU free  memory = % .2f MB\n", (float)free / (1.024e6));
	printf("Total device number = :%d\n\n", n_dev);
	for (int i = 0; i < NUM_GPU; i++) {
		dev_id[i] = i%n_dev;
		assert(dev_id[0] == dev_num);
	}
	for (int i = 0; i < n_dev; i++) {
		cudaDeviceEnablePeerAccess(i, 0);
	}
	int accessibleTest;
	cudaDeviceCanAccessPeer(&accessibleTest, dev_id[0], dev_id[1]);
	if (accessibleTest != 1) { std::cerr << "peer access not supported" << std::endl; };
}

__host__ int allocDeviceMem(problem&  pb) {
	
	cudaError_t err;
	pb.extent = make_cudaExtent(
		2*(pb.mx/2+1) * sizeof(REAL), pb.my, pb.mz);

	pb.tExtent = make_cudaExtent(
		pb.mz * sizeof(cuRPCF::complex), pb.nx/2+1, pb.ny);

	pb.pExtent = make_cudaExtent(
		2 * (pb.mx / 2 + 1) * sizeof(REAL), pb.my, pb.pz);
	
//	cudaExtent & extent = pb.extent;
	cudaExtent & tExtent = pb.tExtent;
	cudaExtent & pExtent = pb.pExtent;

	// Get pitch value of the pointer.
	err = cudaMalloc3D(&(pb.dptr_tu), tExtent);
	pb.tPitch = pb.dptr_tu.pitch;
	safeCudaFree(pb.dptr_tu.ptr);
	pb.dptr_tu.ptr = nullptr;

	initMyCudaMalloc(dim3(pb.mx, pb.my, pb.mz));

	//cuCheck(cudaMalloc3D(&(pb.dptr_u), pExtent),"allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_v), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_w), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_omega_x), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_omega_y), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_omega_z), pExtent), "allocate");

	cuCheck(myCudaMalloc(pb.dptr_u, XYZ_3D), "allocate");
	cuCheck(myCudaMalloc(pb.dptr_v, XYZ_3D), "allocate");
	cuCheck(myCudaMalloc(pb.dptr_w, XYZ_3D), "allocate");
	cuCheck(myCudaMalloc(pb.dptr_omega_x, XYZ_3D), "allocate");
	cuCheck(myCudaMalloc(pb.dptr_omega_y, XYZ_3D), "allocate");
	cuCheck(myCudaMalloc(pb.dptr_omega_z, XYZ_3D), "allocate");

	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_x), extent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_y), extent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_z), extent), "allocate");

	pb.tSize = pb.tPitch * (pb.nx / 2 + 1) * pb.ny;
//	size_t& tsize = pb.tSize;
	//pb.nonlinear_v = (cuRPCF::complex*)malloc(tsize);
	//pb.nonlinear_v_p = (cuRPCF::complex*)malloc(tsize);
	//pb.nonlinear_omega_y = (cuRPCF::complex*)malloc(tsize);
	//pb.nonlinear_omega_y_p = (cuRPCF::complex*)malloc(tsize);
	//ASSERT(pb.nonlinear_v != nullptr);
	//ASSERT(pb.nonlinear_v_p != nullptr);
	//ASSERT(pb.nonlinear_omega_y != nullptr);
	//ASSERT(pb.nonlinear_omega_y_p != nullptr);

	//err = cudaMalloc3D(&(pb.dptr_tv), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tw), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_x), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_y), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_z), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_x), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_y), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_z), tExtent);
	pb.dptr_tu.ptr = nullptr;
	pb.dptr_tv.ptr = nullptr;
	pb.dptr_tw.ptr = nullptr;
	pb.dptr_tomega_x.ptr = nullptr;
	pb.dptr_tomega_y.ptr = nullptr;
	pb.dptr_tomega_z.ptr = nullptr;
	pb.dptr_lamb_x.ptr = nullptr;
	pb.dptr_lamb_y.ptr = nullptr;
	pb.dptr_lamb_z.ptr = nullptr;
	pb.dptr_tLamb_x.ptr = nullptr;
	pb.dptr_tLamb_y.ptr = nullptr;
	pb.dptr_tLamb_z.ptr = nullptr;

	pb.pitch = pb.dptr_u.pitch; 
	pb.size = pb.pitch * pb.my * pb.mz;
	pb.pSize = pb.pitch * pb.my * pb.pz;

	ASSERT(!err);

	return 0;
}

// note : x and y should be normalized by lx and ly.
// i.e. x = x/lx
#define EPSILON_INIT 0.005


// MARK :: this part is the modified CORRECT initial condition, remove comment mark before use
// ____________________________ BEGIN________________________//
//__device__ REAL _get_init_u(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL PI = 4*atan(1.0);
//	return EPSILON_INIT*lx*sin(PI*z)
//		*(cos(2 * PI*x / lx)*sin(2.0*PI*y / ly)
//			+ 0.5*cos(4.0*PI*x / lx)*sin(2 * PI*y / ly)
//			+ cos(2 * PI*x / lx)*sin(4 * PI*y / ly));
//	//return sin(PI*x)*cos(2*PI*y);
//	//return (-2.0 / 3.0 *lx *(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x)
//	//	*sin(2.0*PI*y) + sin(4.0*PI*x)
//	//	*sin(2.0*PI*y) + sin(2.0*PI*x)
//	//	*sin(4.0*PI*y)));
//}
//
//__device__ REAL _get_init_v(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL PI = 4 * atan(1.0);
//	return -EPSILON_INIT*ly*sin(PI*z)
//		*(0.5*sin(2 * PI*x / lx)*cos(2.0*PI*y / ly)
//			+ 0.5*sin(4.0*PI*x / lx)*cos(2.0 * PI*y / ly)
//			+ 0.25*sin(2.0 * PI*x / lx)*cos(4.0 * PI*y / ly));
//	//return -2.00 / 3.0*(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x) 
////		*sin(2.0*PI*y) + sin(4.0*PI*x) 
//	//	*sin(2.0*PI*y) + sin(2.0*PI*x) 
//	//	*sin(4.0*PI*y));
//}
//
//__device__ REAL _get_init_w(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL PI = 4 * atan(1.0);
//	return EPSILON_INIT*(-1.0)*(1.0+cos(PI*z))
//		*(sin(2*PI*x/lx)*sin(2*PI*y/ly)
//			+sin(4*PI*x/lx)*sin(2*PI*y/ly)
//			+sin(2*PI*x/lx)*sin(4*PI*y/ly));
//
//	//return -ly*sin(1.5*PI*z)*(0.5*sin(2.0*PI*x) 
//	//	*cos(2.0*PI*y) + 0.5*sin(4.0*PI*x) 
//	//	*cos(2.0*PI*y) + 0.25*sin(2.0*PI*x) 
//	//	*cos(4.0*PI*y));
//}
//
//__device__ REAL _get_init_omegax(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL pi = 4 * atan(1.0);
//	return (-EPSILON_INIT*ly*pi*cos(pi*z)*(0.5*sin(2.0*pi*x/lx) 
//		*cos(2.0*pi*y/ly) + 0.5*sin(4.0*pi*x/lx) 
//		*cos(2.0*pi*y/ly) + 0.25*sin(2.0*pi*x/lx) 
//		*cos(4.0*pi*y/ly)))
//
//		-(EPSILON_INIT*(1.0 + cos(pi*z))*4.0*pi / ly*(0.5*sin(2.0*pi*x/lx) 
//			*cos(2.0*pi*y/ly) + 0.5*sin(4.0*pi*x/lx) 
//			*cos(2.0*pi*y/ly) + sin(2.0*pi*x/lx) 
//			*cos(4.0*pi*y/ly)));
//}
//
//__device__ REAL _get_init_omegaz(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL pi = 4 * atan(1.0);
//	return EPSILON_INIT*2.0*pi*sin(pi*z)*  
//		(lx / ly*(cos(2.0*pi*x/lx)*cos(2.0*pi*y/ly) 
//			+0.5*cos(4.0*pi*x/lx)*cos(2.0*pi*y/ly) 
//			+2.0*cos(2.0*pi*x/lx)*cos(4.0*pi*y/ly)) 
//			+
//			ly / lx*(0.5*cos(2.0*pi*x/lx)*cos(2.0*pi*y/ly) 
//				+cos(4.0*pi*x/lx)*cos(2.0*pi*y/ly) 
//				+0.25*cos(2.0*pi*x/lx)*cos(4.0*pi*y/ly)));
//}
//
//
//__device__ REAL _get_init_omegay(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
//	const REAL PI = 4 * atan(1.0);
//	return
//		EPSILON_INIT*(-1.0) *(1.0 + cos(PI*z))
//		*2*PI/lx*(
//			     cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
//			+2.0*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
//			+    cos(2 * PI*x / lx)*sin(4 * PI*y / ly))
//		-
//		EPSILON_INIT*lx*PI*cos(PI*z)*(
//		      cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
//		+ 0.5*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
//		+     cos(2 * PI*x / lx)*sin(4 * PI*y / ly)
//		);
//}
//
//
//_____________________________END_______________________________


__device__ REAL _get_init_u(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL PI = 4 * atan(1.0);
	return EPSILON_INIT*lx*sin(1.5*PI*z)
		*(cos(2 * PI*x / lx)*sin(2.0*PI*y / ly)
			+ 0.5*cos(4.0*PI*x / lx)*sin(2 * PI*y / ly)
			+ cos(2 * PI*x / lx)*sin(4 * PI*y / ly));
	//return sin(PI*x)*cos(2*PI*y);
	//return (-2.0 / 3.0 *lx *(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x)
	//	*sin(2.0*PI*y) + sin(4.0*PI*x)
	//	*sin(2.0*PI*y) + sin(2.0*PI*x)
	//	*sin(4.0*PI*y)));
}

__device__ REAL _get_init_v(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL PI = 4 * atan(1.0);
	return -EPSILON_INIT*ly*sin(1.5*PI*z)
		*(0.5*sin(2 * PI*x / lx)*cos(2.0*PI*y / ly)
			+ 0.5*sin(4.0*PI*x / lx)*cos(2.0 * PI*y / ly)
			+ 0.25*sin(2.0 * PI*x / lx)*cos(4.0 * PI*y / ly));
	//return -2.00 / 3.0*(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x) 
	//		*sin(2.0*PI*y) + sin(4.0*PI*x) 
	//	*sin(2.0*PI*y) + sin(2.0*PI*x) 
	//	*sin(4.0*PI*y));
}

__device__ REAL _get_init_w(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL PI = 4 * atan(1.0);
	return EPSILON_INIT*(-2.0/3.0)*(1.0 + cos(1.5*PI*z))
		*(sin(2 * PI*x / lx)*sin(2 * PI*y / ly)
			+ sin(4 * PI*x / lx)*sin(2 * PI*y / ly)
			+ sin(2 * PI*x / lx)*sin(4 * PI*y / ly));

	//return -ly*sin(1.5*PI*z)*(0.5*sin(2.0*PI*x) 
	//	*cos(2.0*PI*y) + 0.5*sin(4.0*PI*x) 
	//	*cos(2.0*PI*y) + 0.25*sin(2.0*PI*x) 
	//	*cos(4.0*PI*y));
}

__device__ REAL _get_init_omegax(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL pi = 4 * atan(1.0);
	return (-EPSILON_INIT*ly*1.5*pi*cos(1.5*pi*z)*(0.5*sin(2.0*pi*x / lx)
		*cos(2.0*pi*y / ly) + 0.5*sin(4.0*pi*x / lx)
		*cos(2.0*pi*y / ly) + 0.25*sin(2.0*pi*x / lx)
		*cos(4.0*pi*y / ly)))

		- (2.0/3.0*EPSILON_INIT*(1.0 + cos(1.5*pi*z))*4.0*pi / ly*(0.5*sin(2.0*pi*x / lx)
			*cos(2.0*pi*y / ly) + 0.5*sin(4.0*pi*x / lx)
			*cos(2.0*pi*y / ly) + sin(2.0*pi*x / lx)
			*cos(4.0*pi*y / ly)));
}

__device__ REAL _get_init_omegaz(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL pi = 4 * atan(1.0);
	return EPSILON_INIT*2.0*pi*sin(1.5*pi*z)*
		(lx / ly*(cos(2.0*pi*x / lx)*cos(2.0*pi*y / ly)
			+ 0.5*cos(4.0*pi*x / lx)*cos(2.0*pi*y / ly)
			+ 2.0*cos(2.0*pi*x / lx)*cos(4.0*pi*y / ly))
			+
			ly / lx*(0.5*cos(2.0*pi*x / lx)*cos(2.0*pi*y / ly)
				+ cos(4.0*pi*x / lx)*cos(2.0*pi*y / ly)
				+ 0.25*cos(2.0*pi*x / lx)*cos(4.0*pi*y / ly)));
}


__device__ REAL _get_init_omegay(REAL x, REAL y, REAL z, REAL lx, REAL ly) {
	const REAL PI = 4 * atan(1.0);
	return
		EPSILON_INIT*(-1.0) *(1.0 + cos(1.5*PI*z))
		* 2 * PI / lx*(
			cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
			+ 2.0*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
			+ cos(2 * PI*x / lx)*sin(4 * PI*y / ly))
		-
		EPSILON_INIT*lx*PI*1.5*cos(1.5*PI*z)*(
			cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
			+ 0.5*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
			+ cos(2 * PI*x / lx)*sin(4 * PI*y / ly)
			);
}

// compute initial flow, save the data to pointer defined in pb.
// assuming the pointer are already initialized by initCUDA.
__global__ void init_flow_kernel(
	REAL* dptr_u, REAL* dptr_v, REAL* dptr_w, 
	REAL* dptr_ox, REAL* dptr_oy, REAL* dptr_oz, 
	REAL lx, REAL ly,
	int px, int py, int pz, int pitch) {

	int y = threadIdx.x + blockDim.x*blockIdx.x;
	int z = threadIdx.y + blockDim.y*blockIdx.y;

	if (y >= py || z >= pz) return;

	const REAL pi = 4 * atan(1.0);

	REAL xx, yy, zz;
	REAL* u_row, *v_row, *w_row, *ox_row, *oy_row, *oz_row;
	//ASSERT(pitch > 0);
	//ASSERT(dptr_u!=nullptr);

	size_t inc = pitch*(py*z + y)/sizeof(REAL);

	u_row = dptr_u + inc;
	v_row = dptr_v + inc;
	w_row = dptr_w + inc;
	ox_row = dptr_ox + inc;
	oy_row = dptr_oy + inc;
	oz_row = dptr_oz + inc;

	if (z == 0 || z == pz - 1) {
		for (int x = 0; x < px; x++) {
			u_row[x] = 0.0;
			v_row[x] = 0.0;
			w_row[x] = 0.0;
			ox_row[x] = 0.0;
			oy_row[x] = 0.0;
			oz_row[x] = 0.0;
		}
	}
	else 
	{
		for (int x = 0; x < px; x++) {

			xx = (x*1.0) / px * lx;
			yy = (y*1.0) / py * ly;
			zz = cos(pi*z / (pz - 1));
			u_row[x] = _get_init_u(xx, yy, zz, lx, ly);
			v_row[x] = _get_init_v(xx, yy, zz, lx, ly);
			w_row[x] = _get_init_w(xx, yy, zz, lx, ly);
			ox_row[x] = _get_init_omegax(xx, yy, zz, lx, ly);
			oy_row[x] = _get_init_omegay(xx, yy, zz, lx, ly);
			oz_row[x] = _get_init_omegaz(xx, yy, zz, lx, ly);
		}
	}

}

__host__ int initFlow(problem& pb) {

	cudaError_t err = cudaDeviceSynchronize(); // CudaErrorLaunchFailure
	ASSERT(err == cudaSuccess);

	//int nthreadx = 16;
	//int nthready = 16;
	//int nDimx = pb.py / nthreadx;
	//int nDimy = pb.pz / nthready;
	//if (pb.py % nthreadx != 0) nDimx++;
	//if (pb.pz % nthready != 0) nDimy++;
	//dim3 nThread(nthreadx, nthready);
	//dim3 nDim(nDimx, nDimy);

	init_flow_kernel <<<pb.npDim, pb.nThread>>> ((REAL*)pb.dptr_u.ptr,
		(REAL*)pb.dptr_v.ptr,		(REAL*)pb.dptr_w.ptr, 
		(REAL*)pb.dptr_omega_x.ptr,	(REAL*)pb.dptr_omega_y.ptr,
		(REAL*)pb.dptr_omega_z.ptr,
		pb.lx, pb.ly, pb.px, pb.py, pb.nz, pb.dptr_u.pitch);
	//system("pause");
	err = cudaDeviceSynchronize(); // CudaErrorLaunchFailure
	ASSERT(err == cudaSuccess);

	REAL* buffer;
	size_t& size = pb.pSize; //pb.dptr_u.pitch*pb.my*pb.mz;
	size_t& tSize = pb.tSize;// pb.tPitch*(pb.mx / 2 + 1)*pb.my;

	//buffer = (REAL*)malloc(size);
	//cuCheck(cudaMemcpy(buffer, pb.dptr_u.ptr, size, cudaMemcpyDeviceToHost),"memcpy");
	//err = cudaDeviceSynchronize();
	//ASSERT(err == cudaSuccess);
	//RPCF::write_3d_to_file("init.txt", buffer, pb.dptr_u.pitch, (pb.mx), pb.my, pb.pz);
	

	int dim[3];
	dim[0] = pb.mx;
	dim[1] = pb.my;
	dim[2] = pb.mz;

	int tDim[3];
	tDim[0] = pb.mz;
	tDim[1] = pb.mx;
	tDim[2] = pb.my;
	
	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim, No_Padding);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim, No_Padding);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim, No_Padding);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim, No_Padding);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim, No_Padding);
	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim, No_Padding);
	
	//copy initial rhs_v and rhs_omeag_y
	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	
	getUVW(pb);

	for (int k = 0; k < pb.nz; k++) {
		pb.tv0[k] = pb.rhs_v[k];
		pb.tomega_y_0[k] = pb.rhs_omega_y[k];
	}

	for (int j = 0; j < pb.ny; j++) {
		for (int i = 0; i < (pb.nx / 2 + 1); i++) {
			for (int k = 0; k < pb.mz; k++) {
				size_t inc = k+pb.tPitch/sizeof(cuRPCF::complex)*(j*(pb.nx / 2 + 1) + i);
				pb.rhs_v_p[inc] = pb.rhs_v[inc];
			}
		}
	}

	//safeFree(buffer);
	return 0;
}
//
//__host__ int computeNonlinear(problem& pb) {
//
//	return 0;
//}


__host__ __device__ void ddz(REAL* u, int N) {
	REAL buffer[MAX_NZ*4];
	REAL dmat;
	for (int i = 0; i < N; i++) {
		buffer[i] = 0;
		for (int j = i+1; j < N; j=j+2) {
			dmat = 2 * (j);
			buffer[i] = buffer[i] + dmat * u[j];
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}

__host__ __device__ void ddz(cuRPCF::complex *u, int N) {
	cuRPCF::complex buffer[MAX_NZ];
	REAL dmat;
	cuRPCF::complex buffer_u[MAX_NZ];
	for (int i = 0; i < N; i++) {
		buffer_u[i] = u[i];
	}
	for (int i = 0; i < N; i++) {
		buffer[i] = cuRPCF::complex(0.0,0.0);
		for (int j = i + 1; j < N; j = j + 2) {
			dmat = 2 * REAL(j);
			buffer[i] = buffer[i] + buffer_u[j] * dmat;
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}


__device__ void ddz_sm(REAL* u, int N, int kz) {
	REAL buffer;
	REAL dmat;
	
	//wait all threads to load data before computing
	__syncthreads();

	buffer = 0.0;
	for (int j = kz + 1; j < N; j = j + 2) {
		dmat = 2 * REAL(j);
		buffer = buffer + u[j] * dmat;
	}
	//wait all threads to finish computation before overwriting array.
	__syncthreads();
	if (kz == 0) {
		u[0] = buffer * 0.5;
	}
	else
	{
		u[kz] = buffer;
	}
}

__device__ void ddz_sm(cuRPCF::complex *u, int N, int kz) {
	cuRPCF::complex buffer;
	REAL dmat;

	//wait all threads to load data before computing
	__syncthreads();

	buffer = cuRPCF::complex(0.0,0.0);
	for (int j = kz + 1; j < N; j = j + 2) {
		dmat = 2 * REAL(j);
		buffer = buffer + u[j] * dmat;
	}	
	//wait all threads to finish computation before overwriting array.
	__syncthreads();
	if (kz == 0) {
		u[0] = buffer * 0.5;
	}
	else
	{
		u[kz] = buffer;
	}
}

__host__ __device__
void get_ialpha_ibeta(int kx, int ky, int ny,
	REAL alpha, REAL beta,
	REAL& ialpha, REAL& ibeta )
{
	ialpha = (REAL)kx / alpha;
	ibeta = (REAL)ky / beta;
	if (ky >= ny / 2 + 1) {
		ibeta = REAL(ky - ny) / beta;
	}
}

// This kernel function is used to perform multiply between matrix and vector;
__global__
void m_multi_v_kernel(cuRPCF::complex* _mat, cuRPCF::complex* _v, const int N, const size_t pitch) {
	const int iMat = blockIdx.x;
	const int J = threadIdx.x;
	const int tid = J;
	__shared__ cuRPCF::complex UI[MAX_NZ];
	__shared__ cuRPCF::complex buffer[MAX_NZ];
	cuRPCF::complex* mat = _mat + iMat*N*N + J*N;
	cuRPCF::complex* v = _v + pitch / sizeof(cuRPCF::complex)*iMat;
	//cuRPCF::complex mat_cache[MAX_NZ];
	//cuRPCF::complex v_cache[MAX_NZ];

	//for (int i = 0; i < N; i++) {
	//	mat_cache[i] = mat[i];
	//}
	//for (int i = 0; i < N; i++) {
	//	v_cache[i] = v[i];
	//}
	//cuRPCF::complex res = cuRPCF::complex(0.0, 0.0);
	//for (int k = 0; k < N; k++) {
	//	res = res + mat_cache[k] * v_cache[k];
	//}

	cuRPCF::complex res[MAX_NZ];
	__shared__ cuRPCF::complex reduction[MAX_NZ];
	// for each row
	for (int i = 0; i < N; i++) {
		UI[J] = mat[i*N + J];
		buffer[J] = v[J];
		__syncthreads();
		buffer[J] = UI[J] * buffer[J];
		__syncthreads();
		if (tid == 0 && N % 2 != 0) buffer[tid] = buffer[tid] + buffer[N - 1];
		for (int s = N/2; s>0; s = s / 2)
		{
			if (tid < s) buffer[tid] = buffer[tid] + buffer[tid + s];
			if (tid == 0 && s % 2 != 0) buffer[tid] = buffer[tid] + buffer[s-1];
			__syncthreads();
		}
		res[i] = buffer[0];
	}
	__syncthreads();
	v[J] = res[J];

	//cuRPCF::complex res[MAX_NZ];
	////cuRPCF::complex* temp = (cuRPCF::complex*)malloc(N*sizeof(cuRPCF::complex));
	
	//for (int i = 0; i < N; i++) {
	//	UI[J] = mat[i*N + J];
	//	__syncthreads();

	//	buffer[J] = 0;
	//	buffer[J] = UI[J] * v[J];
	//	if(J == 0){
	//		for (int j = 1; j < N; j++) {
	//			buffer[0] = buffer[0] + buffer[j];
	//		}
	//		res[i] = buffer[0];
	//	}
	//}
	//__syncthreads();
	//v[J] = res[J];
}


__host__ cudaError_t m_multi_v_gpu(cuRPCF::complex* _mat, cuRPCF::complex* v, const int N, const size_t pitch, const int batch) {
	m_multi_v_kernel <<<batch, N >>>(_mat, v, N, pitch);
	return cudaDeviceSynchronize();
}


