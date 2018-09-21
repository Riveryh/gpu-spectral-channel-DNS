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
	cudaGetDevice(&dev_num);
	cudaGetDeviceProperties(&prop, dev_num);
	cudaMemGetInfo(&free, &total);
	//err = cudaDeviceReset();
	//ASSERT(err == cudaSuccess);
	printf("\nUsing CUDA device %u. Device ID: %s on PCI-E %d\n",
		dev_num, prop.name, prop.pciBusID); 
	printf("\nGPU total memory = % .2f MB\n", (float)total / (1.024e6));
	printf("\nGPU free  memory = % .2f MB\n", (float)free / (1.024e6));

	cudaExtent extent = make_cudaExtent(
		2*(pb.mx/2+1) * sizeof(real), pb.my, pb.mz);

	cudaExtent tExtent = make_cudaExtent(
		pb.mz * sizeof(complex), pb.mx/2+1, pb.my);


	// Get pitch value of the pointer.
	err = cudaMalloc3D(&(pb.dptr_tu), tExtent);
	pb.tPitch = pb.dptr_tu.pitch;
	cudaFree(pb.dptr_tu.ptr);
	pb.dptr_tu.ptr = nullptr;

	cuCheck(cudaMalloc3D(&(pb.dptr_u), extent),"allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_v), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_w), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_omega_x), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_omega_y), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_omega_z), extent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_x), extent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_y), extent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_z), extent), "allocate");

	size_t tsize = pb.tPitch * (pb.mx / 2 + 1) * pb.my;
	pb.nonlinear_v = (complex*)malloc(tsize);
	pb.nonlinear_v_p = (complex*)malloc(tsize);
	pb.nonlinear_omega_y = (complex*)malloc(tsize);
	pb.nonlinear_omega_y_p = (complex*)malloc(tsize);
	ASSERT(pb.nonlinear_v != nullptr);
	ASSERT(pb.nonlinear_v_p != nullptr);
	ASSERT(pb.nonlinear_omega_y != nullptr);
	ASSERT(pb.nonlinear_omega_y_p != nullptr);

	//err = cudaMalloc3D(&(pb.dptr_tv), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tw), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_x), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_y), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tomega_z), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_x), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_y), tExtent);
	//err = cudaMalloc3D(&(pb.dptr_tLamb_z), tExtent);

	pb.pitch = pb.dptr_u.pitch; 


	ASSERT(!err);

	return 0;
}

// note : x and y should be normalized by lx and ly.
// i.e. x = x/lx
__device__ inline real _get_init_u(real x, real y, real z, real lx, real ly) {
	const real PI = 4*atan(1.0);
	//return sin(PI*x)*cos(2*PI*y);
	return (-2.0 / 3.0 *lx *(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x)
		*sin(2.0*PI*y) + sin(4.0*PI*x)
		*sin(2.0*PI*y) + sin(2.0*PI*x)
		*sin(4.0*PI*y)));
}

__device__ inline real _get_init_v(real x, real y, real z, real lx, real ly) {
	const real PI = 4 * atan(1.0);
	return -2.00 / 3.0*(1.0 + cos(1.5*PI*z))*(sin(2.0*PI*x) 
		*sin(2.0*PI*y) + sin(4.0*PI*x) 
		*sin(2.0*PI*y) + sin(2.0*PI*x) 
		*sin(4.0*PI*y));
}

__device__ inline real _get_init_w(real x, real y, real z, real lx, real ly) {
	const real PI = 4 * atan(1.0);
	return -ly*sin(1.5*PI*z)*(0.5*sin(2.0*PI*x) 
		*cos(2.0*PI*y) + 0.5*sin(4.0*PI*x) 
		*cos(2.0*PI*y) + 0.25*sin(2.0*PI*x) 
		*cos(4.0*PI*y));
}

__device__ inline real _get_init_omegax(real x, real y, real z, real lx, real ly) {
	const real pi = 4 * atan(1.0);
	return (-ly*1.5*pi*cos(1.5*pi*z)*(0.5*sin(2.0*pi*x) 
		*cos(2.0*pi*y) + 0.5*sin(4.0*pi*x) 
		*cos(2.0*pi*y) + 0.25*sin(2.0*pi*x) 
		*cos(4.0*pi*y)))

		-(2.0 / 3.0*(1.0 + cos(1.5*pi*z))*4.0*pi / ly*(0.5*sin(2.0*pi*x) 
			*cos(2.0*pi*y) + 0.5*sin(4.0*pi*x) 
			*cos(2.0*pi*y) + sin(2.0*pi*x) 
			*cos(4.0*pi*y)));
}

__device__ inline real _get_init_omegay(real x, real y, real z, real lx, real ly) {
	const real pi = 4 * atan(1.0);
	return 2.0*pi*sin(1.5*pi*z)*  
		(lx / ly*(cos(2.0*pi*x)*cos(2.0*pi*y) 
			+0.5*cos(4.0*pi*x)*cos(2.0*pi*y) 
			+2.0*cos(2.0*pi*x)*cos(4.0*pi*y)) 
			+
			ly / lx*(0.5*cos(2.0*pi*x)*cos(2.0*pi*y) 
				+cos(4.0*pi*x)*cos(2.0*pi*y) 
				+0.25*cos(2.0*pi*x)*cos(4.0*pi*y)));
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

	if (y >= py || y >= pz) return;

	const real pi = 4 * atan(1.0);

	real xx, yy, zz;
	real* u_row, *v_row, *w_row, *ox_row, *oy_row;
	//ASSERT(pitch > 0);
	//ASSERT(dptr_u!=nullptr);

	size_t inc = pitch*(py*z + y)/sizeof(real);

	u_row = dptr_u + inc;
	v_row = dptr_v + inc;
	w_row = dptr_w + inc;
	ox_row = dptr_ox + inc;
	oy_row = dptr_oy + inc;
	
	for (int x = 0; x < px; x++) {
		xx = (x*1.0) / px;
		yy = (y*1.0) / py;
		zz = cos(pi*z / (pz - 1));
		u_row[x] = _get_init_u(xx, yy, zz, lx, ly);
		v_row[x] = _get_init_v(xx, yy, zz, lx, ly);
		w_row[x] = _get_init_w(xx, yy, zz, lx, ly);
		ox_row[x] = _get_init_omegax(xx, yy, zz, lx, ly);
		oy_row[x] = _get_init_omegay(xx, yy, zz, lx, ly);
	}
}

__host__ int initFlow(problem& pb) {

	cudaError_t err = cudaDeviceSynchronize(); // CudaErrorLaunchFailure
	ASSERT(err == cudaSuccess);

	int nthreadx = 16;
	int nthready = 16;
	int nDimx = pb.py / nthreadx;
	int nDimy = pb.pz / nthready;
	if (pb.py % nthreadx != 0) nDimx++;
	if (pb.pz % nthready != 0) nDimy++;
	dim3 nThread(nthreadx, nthready);
	dim3 nDim(nDimx, nDimy);

	init_flow_kernel <<<nDim, nThread>>> ((real*)pb.dptr_u.ptr,
		(real*)pb.dptr_v.ptr,		(real*)pb.dptr_w.ptr, 
		(real*)pb.dptr_omega_x.ptr,	(real*)pb.dptr_omega_y.ptr,
		(real*)pb.dptr_omega_z.ptr,
		pb.lx, pb.ly, pb.px, pb.py, pb.pz, pb.dptr_u.pitch);
	//system("pause");
	err = cudaDeviceSynchronize(); // CudaErrorLaunchFailure
	ASSERT(err == cudaSuccess);

	real* buffer;
	size_t size = pb.dptr_u.pitch*pb.my*pb.mz;
	size_t tSize = pb.tPitch*(pb.mx/2+1)*pb.my;

	buffer = (real*)malloc(size);
	cuCheck(cudaMemcpy(buffer, pb.dptr_u.ptr, size, cudaMemcpyDeviceToHost),"memcpy");
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);
	//RPCF::write_3d_to_file("init.txt", buffer, pb.dptr_u.pitch, (pb.mx), pb.my, pb.mz);
	

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

	cuCheck(cudaMemcpy(pb.rhs_v, pb.dptr_tw.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.rhs_omega_y, pb.dptr_tomega_z.ptr, tSize, cudaMemcpyDeviceToHost), "memcpy");
	
	for (int j = 0; j < pb.my; j++) {
		for (int i = 0; i < (pb.mx / 2 + 1); i++) {
			for (int k = 0; k < pb.mz; k++) {
				size_t inc = pb.tPitch/sizeof(complex)*(j*(pb.mx / 2 + 1) + i) + k;
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
	complex buffer[MAX_NZ * 4];
	complex dmat;
	for (int i = 0; i < N; i++) {
		buffer[i] = 0;
		for (int j = i + 1; j < N; j = j + 2) {
			dmat = 2 * real(j);
			buffer[i] = buffer[i] + dmat * u[j];
		}
	}
	u[0] = buffer[0] * 0.5;
	for (int i = 1; i < N; i++) {
		u[i] = buffer[i];
	}
}

