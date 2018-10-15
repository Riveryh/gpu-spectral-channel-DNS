#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data.h"
#include "RPCFKernels.cuh"
#include "transform.cuh"
#include "operation.h"
#include "cuRPCF.h"
#include <cstdlib>
#include <math.h>
#include <cassert>
#include "rhs.cuh"

__global__ void vKernel(cudaPitchedPtr dpPtr,
	int width, int height, int depth) {
	char* dptr = (char*)dpPtr.ptr;
	size_t pitch = dpPtr.pitch;
	size_t slicePitch = pitch*height;
	for (int z = 0; z < depth; ++z) {
		char* slice = dptr + z * slicePitch;
		for (int y = 0; y < height; ++y) {
			real* row = (real*)(slice + y * pitch);
			for (int x = 0; x < width; ++x) {
				row[x] = 100.0*x + 10.0*y + 1.0*z;
			}
			printf("%d,%d,%d,%10.3f\n", width, height, depth, row[0]);
		}
	}
}

__host__ int initCUDA(problem&  pb) {
	cudaDeviceProp prop;
	int dev_num;
	cudaError_t err;
	size_t free;
	size_t total;
	init_pthread(pb);
	cudaGetDevice(&dev_num);
	cudaGetDeviceProperties(&prop, dev_num);
	cudaMemGetInfo(&free, &total);
	//err = cudaDeviceReset();
	//ASSERT(err == cudaSuccess);
	printf("\nUsing CUDA device %u. Device ID: %s on PCI-E %d\n",
		dev_num, prop.name, prop.pciBusID); 
	printf("\nGPU total memory = % .2f MB\n", (float)total / (1.024e6));
	printf("\nGPU free  memory = % .2f MB\n", (float)free / (1.024e6));

	pb.extent = make_cudaExtent(
		2*(pb.mx/2+1) * sizeof(real), pb.my, pb.mz);

	pb.tExtent = make_cudaExtent(
		pb.mz * sizeof(complex), pb.mx/2+1, pb.my);

	pb.pExtent = make_cudaExtent(
		2 * (pb.mx / 2 + 1) * sizeof(real), pb.my, pb.pz);
	
	cudaExtent & extent = pb.extent;
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
	size_t& tsize = pb.tSize;
	//pb.nonlinear_v = (complex*)malloc(tsize);
	//pb.nonlinear_v_p = (complex*)malloc(tsize);
	//pb.nonlinear_omega_y = (complex*)malloc(tsize);
	//pb.nonlinear_omega_y_p = (complex*)malloc(tsize);
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
__device__ real _get_init_u(real x, real y, real z, real lx, real ly) {
	const real PI = 4*atan(1.0);
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

__device__ real _get_init_v(real x, real y, real z, real lx, real ly) {
	const real PI = 4 * atan(1.0);
	return -EPSILON_INIT*ly*sin(1.5*PI*z)
		*(0.5*sin(2 * PI*x / lx)*cos(2.0*PI*y / ly)
			+ 0.5*sin(4.0*PI*x / lx)*cos(2.0 * PI*y / ly)
			+ 0.25*sin(2.0 * PI*x / lx)*cos(4.0 * PI*y / ly));
	//return -2.00 / 3.0*(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x) 
//		*sin(2.0*PI*y) + sin(4.0*PI*x) 
	//	*sin(2.0*PI*y) + sin(2.0*PI*x) 
	//	*sin(4.0*PI*y));
}

__device__ real _get_init_w(real x, real y, real z, real lx, real ly) {
	const real PI = 4 * atan(1.0);
	return EPSILON_INIT*(-2.0)/3.0*(1.0+cos(1.5*PI*z))
		*(sin(2*PI*x/lx)*sin(2*PI*y/ly)
			+sin(4*PI*x/lx)*sin(2*PI*y/ly)
			+sin(2*PI*x/lx)*sin(4*PI*y/ly));

	//return -ly*sin(1.5*PI*z)*(0.5*sin(2.0*PI*x) 
	//	*cos(2.0*PI*y) + 0.5*sin(4.0*PI*x) 
	//	*cos(2.0*PI*y) + 0.25*sin(2.0*PI*x) 
	//	*cos(4.0*PI*y));
}

__device__ real _get_init_omegax(real x, real y, real z, real lx, real ly) {
	const real pi = 4 * atan(1.0);
	return (-EPSILON_INIT*ly*1.5*pi*cos(1.5*pi*z)*(0.5*sin(2.0*pi*x/lx) 
		*cos(2.0*pi*y/ly) + 0.5*sin(4.0*pi*x/lx) 
		*cos(2.0*pi*y/ly) + 0.25*sin(2.0*pi*x/lx) 
		*cos(4.0*pi*y/ly)))

		-(2.0 / 3.0*EPSILON_INIT*(1.0 + cos(1.5*pi*z))*4.0*pi / ly*(0.5*sin(2.0*pi*x/lx) 
			*cos(2.0*pi*y/ly) + 0.5*sin(4.0*pi*x/lx) 
			*cos(2.0*pi*y/ly) + sin(2.0*pi*x/lx) 
			*cos(4.0*pi*y/ly)));
}

__device__ real _get_init_omegaz(real x, real y, real z, real lx, real ly) {
	const real pi = 4 * atan(1.0);
	return EPSILON_INIT*2.0*pi*sin(1.5*pi*z)*  
		(lx / ly*(cos(2.0*pi*x/lx)*cos(2.0*pi*y/ly) 
			+0.5*cos(4.0*pi*x/lx)*cos(2.0*pi*y/ly) 
			+2.0*cos(2.0*pi*x/lx)*cos(4.0*pi*y/ly)) 
			+
			ly / lx*(0.5*cos(2.0*pi*x/lx)*cos(2.0*pi*y/ly) 
				+cos(4.0*pi*x/lx)*cos(2.0*pi*y/ly) 
				+0.25*cos(2.0*pi*x/lx)*cos(4.0*pi*y/ly)));
}


__device__ real _get_init_omegay(real x, real y, real z, real lx, real ly) {
	const real PI = 4 * atan(1.0);
	return
		EPSILON_INIT*(-2.0) / 3.0*(1.0 + cos(1.5*PI*z))
		*2*PI/lx*(
			     cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
			+2.0*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
			+    cos(2 * PI*x / lx)*sin(4 * PI*y / ly))
		-
		EPSILON_INIT*lx*1.5*PI*cos(1.5*PI*z)*(
		      cos(2 * PI*x / lx)*sin(2 * PI*y / ly)
		+ 0.5*cos(4 * PI*x / lx)*sin(2 * PI*y / ly)
		+     cos(2 * PI*x / lx)*sin(4 * PI*y / ly)
		);
}

// compute initial flow, save the data to pointer defined in pb.
// assuming the pointer are already initialized by initCUDA.
__global__ void init_flow_kernel(
	real* dptr_u, real* dptr_v, real* dptr_w, 
	real* dptr_ox, real* dptr_oy, real* dptr_oz, 
	real lx, real ly,
	int px, int py, int pz, int pitch) {

	int y = threadIdx.x + blockDim.x*blockIdx.x;
	int z = threadIdx.y + blockDim.y*blockIdx.y;

	if (y >= py || z >= pz) return;

	const real pi = 4 * atan(1.0);

	real xx, yy, zz;
	real* u_row, *v_row, *w_row, *ox_row, *oy_row, *oz_row;
	//ASSERT(pitch > 0);
	//ASSERT(dptr_u!=nullptr);

	size_t inc = pitch*(py*z + y)/sizeof(real);

	u_row = dptr_u + inc;
	v_row = dptr_v + inc;
	w_row = dptr_w + inc;
	ox_row = dptr_ox + inc;
	oy_row = dptr_oy + inc;
	oz_row = dptr_oz + inc;
	
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

	init_flow_kernel <<<pb.npDim, pb.nThread>>> ((real*)pb.dptr_u.ptr,
		(real*)pb.dptr_v.ptr,		(real*)pb.dptr_w.ptr, 
		(real*)pb.dptr_omega_x.ptr,	(real*)pb.dptr_omega_y.ptr,
		(real*)pb.dptr_omega_z.ptr,
		pb.lx, pb.ly, pb.px, pb.py, pb.pz, pb.dptr_u.pitch);
	//system("pause");
	err = cudaDeviceSynchronize(); // CudaErrorLaunchFailure
	ASSERT(err == cudaSuccess);

	real* buffer;
	size_t& size = pb.pSize; //pb.dptr_u.pitch*pb.my*pb.mz;
	size_t& tSize = pb.tSize;// pb.tPitch*(pb.mx / 2 + 1)*pb.my;

	buffer = (real*)malloc(size);
	//cuCheck(cudaMemcpy(buffer, pb.dptr_u.ptr, size, cudaMemcpyDeviceToHost),"memcpy");
	//err = cudaDeviceSynchronize();
	//ASSERT(err == cudaSuccess);
	//RPCF::write_3d_to_file("init.txt", buffer, pb.dptr_u.pitch, (pb.mx), pb.my, pb.mz);
	

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
	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	
	for (int k = 0; k < pb.nz; k++) {
		pb.tv0[k] = pb.rhs_v[k];
		pb.tomega_y_0[k] = pb.rhs_omega_y[k];
	}

	for (int j = 0; j < pb.ny; j++) {
		for (int i = 0; i < (pb.nx / 2 + 1); i++) {
			for (int k = 0; k < pb.mz; k++) {
				size_t inc = k+pb.tPitch/sizeof(complex)*(j*(pb.nx / 2 + 1) + i);
				pb.rhs_v_p[inc] = pb.rhs_v[inc];
			}
		}
	}

	safeFree(buffer);
	return 0;
}

__host__ int computeNonlinear(problem& pb) {

	return 0;
}


__host__ __device__ void ddz(real* u, int N) {
	real buffer[MAX_NZ*4];
	real dmat;
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

__host__ __device__ void ddz(complex *u, int N) {
	complex buffer[MAX_NZ];
	real dmat;
	for (int i = 0; i < N; i++) {
		buffer[i] = complex(0.0,0.0);
		for (int j = i + 1; j < N; j = j + 2) {
			dmat = 2 * real(j);
			buffer[i] = buffer[i] + u[j] * dmat;
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}

__host__ __device__
void get_ialpha_ibeta(int kx, int ky, int ny,
	real alpha, real beta,
	real& ialpha, real& ibeta )
{
	ialpha = (real)kx / alpha;
	ibeta = (real)ky / beta;
	if (ky >= ny / 2 + 1) {
		ibeta = real(ky - ny) / beta;
	}
}

