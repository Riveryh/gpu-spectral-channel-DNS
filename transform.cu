#include "transform.cuh"
#include <malloc.h>
#include <assert.h>
#include <stdio.h>
#include "operation.h"
#include "cuRPCF.h"
#include <omp.h>
#include "transpose.cuh"

cufftHandle planXYr2c, planXYc2r, planZ_pad, planZ_no_pad;

#define KERNEL_SYNCHRONIZED


__host__ int initFFT(problem &pb) {
	cufftResult res;
	const int mx = pb.mx;
	const int my = pb.my;
	const int mz = pb.mz;
	const int inPitch = pb.pitch;
	const int outPitch = pb.tPitch;
	const int pmx = inPitch / sizeof(real);
	const int pmz = outPitch / sizeof(complex);

	const int istride = 1;
	int inembed[2] = { my, pmx };
	int idist = pmx*my;

	int inembed2[2] = { my,pmx / 2 };

	int idist2 = pmx / 2 * my;

	int dim2[2] = { my,mx };

	int dim1[1] = { mz };
	int onembed[1] = { pmz };
	const int odist = pmz;
	const int ostride = 1;

	int dim1_no_pad[1] = { mz / 2 };

	//cufftPlanMany( plan *, int dim, int* n, int* inembed, int istride, int idist
	//  int* onembed, int ostride, int odist, cufftType, int batch);
	res = cufftPlanMany(&planXYr2c, 2, dim2, inembed, istride, idist,
		inembed2, istride, idist2, myCUFFT_R2C, pb.pz);
	assert(res == CUFFT_SUCCESS);
	res = cufftPlanMany(&planXYc2r, 2, dim2, inembed2, istride, idist2,
		inembed, istride, idist, myCUFFT_C2R, pb.pz);
	assert(res == CUFFT_SUCCESS);
	res = cufftPlanMany(&planZ_pad, 1, dim1, onembed, ostride, odist,
		onembed, ostride, odist, myCUFFT_C2C, (mx/2+1)*my/3);
	assert(res == CUFFT_SUCCESS);
	res = cufftPlanMany(&planZ_no_pad, 1, dim1_no_pad, onembed, ostride, odist,
		onembed, ostride, odist, myCUFFT_C2C, (mx/2+1)*my/3);
	assert(res == CUFFT_SUCCESS);
	return 0;
}



__host__ int transform_3d_one(DIRECTION dir, cudaPitchedPtr& Ptr,
	cudaPitchedPtr& tPtr, int* dim, int* tDim, 
	Padding_mode pd, bool isOutput) {

	//transform in x-y direction
	cufftResult res;

	cudaExtent extent = make_cudaExtent(
	  2*(dim[0]/2+1) * sizeof(real), dim[1], dim[2]);
	cudaError_t err;

	ASSERT(dim[0] == tDim[1]);
	ASSERT(dim[1] == tDim[2]);
	ASSERT(dim[2] == tDim[0]);

	cudaExtent tExtent = make_cudaExtent(
		tDim[0] * sizeof(complex), tDim[1]/2+1 , tDim[2]);

	dim3 threadDim(4, 4);

	real* buffer;
	real* tbuffer;

	// tPtr -> Ptr
	if (dir == BACKWARD) {
		ASSERT(Ptr.ptr == nullptr);
		cuCheck( cudaMalloc3D(&(Ptr), extent),"cuMalloc");

		size_t size = Ptr.pitch*dim[1] * dim[2];
		size_t tSize = tPtr.pitch*(dim[0] / 2 + 1)*dim[1];
		buffer = (real*)malloc(size);
		tbuffer = (real*)malloc(tSize);
		ASSERT(buffer != nullptr);
		ASSERT(tbuffer != nullptr);

		//setZeros <<<1, threadDim >>> (Ptr, dim[0], dim[1], dim[2]);

//#ifdef DEBUG
//		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if(isOutput) RPCF::write_3d_to_file("beforeREV.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
//#endif //DEBUG

		//chebyshev transform in z direction
		cheby_s2p(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);

		//transpose(dir, Ptr, tPtr, dim, tDim);
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);

		int nThreadx = 16;
		int nThready = 16;
		dim3 nThread(nThreadx, nThready);
		int nDimx = dim[1] / nThreadx;
		int nDimy = dim[2] / nThready;
		if (dim[1] % nThreadx != 0) nDimx++;
		if (dim[2] % nThready != 0) nDimy++;
		dim3 nDim(nDimx, nDimy);
		setZeros<<<nDim,nThread>>>((complex*)Ptr.ptr, Ptr.pitch, 
			dim[0], dim[1], dim[2]);
#ifdef KERNEL_SYNCHRONIZED
		cuCheck(cudaDeviceSynchronize(),"set zeros");
#endif
		res = CUFFTEXEC_C2R(planXYc2r, (CUFFTCOMPLEX*)Ptr.ptr,
			(CUFFTREAL*)Ptr.ptr);
		ASSERT(res == CUFFT_SUCCESS);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

//#ifdef DEBUG
//		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if (isOutput) RPCF::write_3d_to_file("afterREV.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
//#endif //DEBUG


//#ifdef DEBUG
//		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if (isOutput) RPCF::write_3d_to_file("afterNORM.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
//#endif //DEBUG


		safeCudaFree(tPtr.ptr);
	}
	else
	{
		// Ptr -> tPtr
		ASSERT(tPtr.ptr == nullptr);
		cuCheck(cudaMalloc3D(&(tPtr), tExtent),"cuMalloc");

		size_t size = Ptr.pitch*dim[1] * dim[2];
		size_t tSize = tPtr.pitch*(dim[0] / 2 + 1)*dim[1];
		buffer = (real*)malloc(size);
		tbuffer = (real*)malloc(tSize);
		ASSERT(buffer != nullptr);
		ASSERT(tbuffer != nullptr);

		//ASSERT(err == cudaSuccess);

//#ifdef DEBUG
//		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if (isOutput) RPCF::write_3d_to_file("before.txt", buffer, Ptr.pitch, 2*(dim[0]/2+1), dim[1], dim[2]);
//#endif //DEBUG

		ASSERT(dir == FORWARD);
		res = CUFFTEXEC_R2C(planXYr2c, (CUFFTREAL*)Ptr.ptr,
			(CUFFTCOMPLEX*)Ptr.ptr);

//#ifdef DEBUG
//		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//
//		if (isOutput) RPCF::write_3d_to_file("afterXY.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
//#endif // DEBUG

		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

		int nthreadx = 16;
		int nthready = 16;
		int nDimx = dim[1] / nthreadx;
		int nDimy = dim[2] / nthready;
		if (dim[1] % nthreadx != 0) nDimx++;
		if (dim[2] % nthready != 0) nDimy++;
		dim3 dim_num(nDimx, nDimy);
		dim3 thread_num(nthreadx, nthready);
		normalize <<< dim_num, thread_num >>>
			(Ptr, dim[0], dim[1], dim[2], 1.0 / dim[0] / dim[1]);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
#endif
		ASSERT(err == cudaSuccess);

		//transpose(FORWARD, Ptr, tPtr, dim, tDim);
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);

		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

		//err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
		//ASSERT(err == cudaSuccess);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);

//#ifdef DEBUG
//		if (isOutput) RPCF::write_3d_to_file("Transposed.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
//#endif //DEBUG

		//transform in z direction
		cheby_p2s(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);

//#ifdef DEBUG
//		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if (isOutput) RPCF::write_3d_to_file("afterZ.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
//#endif //DEBUG

		//setZeros<<<1, threadDim >>>(Ptr, dim[0], dim[1], dim[2]);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);

		safeCudaFree(Ptr.ptr);
	}
	free(buffer);
	free(tbuffer);
	return 0;
}

__global__ void investigate(cudaPitchedPtr p) {
	int i;
	//do something
	i = p.pitch;
	printf("%f",*((real*)p.ptr));
}

__host__ int transform(DIRECTION dir, problem& pb) {
	int indim[3];
	int outdim[3];

	indim[0] = pb.mx;
	indim[1] = pb.my;
	indim[2] = pb.mz;

	outdim[0] = pb.mz;
	outdim[1] = pb.mx;
	outdim[2] = pb.my;

	if (dir == BACKWARD) {
		transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, indim, outdim, Padding);
	}
	if (dir == FORWARD) {
		transform_3d_one(FORWARD, pb.dptr_lamb_x, pb.dptr_tLamb_x, indim, outdim);
		transform_3d_one(FORWARD, pb.dptr_lamb_y, pb.dptr_tLamb_y, indim, outdim);
		transform_3d_one(FORWARD, pb.dptr_lamb_z, pb.dptr_tLamb_z, indim, outdim);
	}
	return 0;
}

//nx, ny, nz is the size of large matrix
//mx, my, mz is the size of the small matrix (dealiased)
__global__ void setZeros(complex* ptr,size_t pitch, int mx, int my, int mz) {
	int ky = threadIdx.x + blockIdx.x*blockDim.x;
	int kz = threadIdx.y + blockIdx.y*blockDim.y;
	if (ky >= my || kz >= mz) return;
	size_t inc = pitch * (kz * my + ky)/sizeof(complex);
	ptr = ptr + inc;
	int nx = mx / 3 * 2;
	int ny = my / 3 * 2;
	
	if (ky >= ny / 2 && ky < ny) {
		for (int ix = 0; ix<mx/2+1; ix++) {
			ptr[ix] = 0.0;
		}
	}
	else
	{
		for (int ix = nx/2+1; ix<mx/2+1; ix++) {
			ptr[ix] = 0.0;
		}
	}
}

__global__ void normalize(cudaPitchedPtr p, int mx, int my, int mz, real factor) {
	const int iy = threadIdx.x + blockIdx.x*blockDim.x;
	const int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= my || iz >= mz/2+1)return;
	if (iy > my / 3  && iy < my / 3 * 2 - 1) return;

	size_t pitch = p.pitch; 
	size_t dist = pitch*(my*iz + iy) / sizeof(real);

	real* row = ((real*)p.ptr) + dist;
	for (int i = 0; i < mx; i++) {
		row[i] = row[i] * factor;
	}
}

//preprocessing of chebyshev transfor, phy to spect
__global__ void cheby_pre_p2s(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;

	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
    
	const int nx = (mx-1) / 3 * 2 + 1;
	const int ny = my / 3 * 2;
	if (ix > nx) return;
	if (iy > ny && iy < my - ny) return;
	
	size_t dist = pitch*(px*iy + ix)/sizeof(complex);
	u = u + dist;
	for (int i = 1; i < pz - 1; i++) {
		u[mz - i].re = u[i].re;
		u[mz - i].im = u[i].im;
	}
}

//preprocessing of chebyshev transform, spect to phy
__global__ void cheby_pre_s2p_pad(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;
	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
	
	const int nx = (mx - 1) / 3 * 2 + 1;
	const int ny = my / 3 * 2;
	if (ix > nx) return;
	if (iy > ny && iy < my - ny) return; 
	
	size_t dist = pitch*(px*iy + ix) / sizeof(complex);
	u = u + dist;
	for (int i = nz; i < pz; i++) {
		u[i].re = 0.0;
		u[i].im = 0.0;
	}
	for (int i = 0; i < nz; i++) {
		u[i].re = u[i].re*0.5;
		u[i].im = u[i].im*0.5;
	}
	for (int i = 1; i < pz - 1; i++) {
		u[mz - i].re = u[i].re;
		u[mz - i].im = u[i].im;
	}
	u[0].re = u[0].re*2.0;
	u[0].im = u[0].im*2.0;
}

__global__ void cheby_pre_s2p_noPad(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;
	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
	
	const int nx = (mx - 1) / 3 * 2 + 1;
	const int ny = my / 3 * 2;
	if (ix > nx) return;
	if (iy > ny && iy < my - ny) return;

	size_t dist = pitch*(px*iy + ix) / sizeof(complex);
	u = u + dist;
	//for (int i = nz; i < pz; i++) {
	//	u[i].re = 0.0;
	//	u[i].im = 0.0;
	//}
	for (int i = 0; i < nz; i++) {
		u[i].re = u[i].re*0.5;
		u[i].im = u[i].im*0.5;
	}
	for (int i = 1; i < nz - 1; i++) {
		u[pz-1 - i].re = u[i].re;
		u[pz-1 - i].im = u[i].im;
	}
	u[0].re = u[0].re*2.0;
	u[0].im = u[0].im*2.0;
}

__global__ void cheby_post_p2s(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
	
	const int nx = (mx - 1) / 3 * 2 + 1;
	const int ny = my / 3 * 2;
	if (ix > nx) return;
	if (iy > ny && iy < my - ny) return;
	
	size_t dist = pitch*(px*iy + ix) / sizeof(complex);
	u = u + dist;
	real factor = (1.0 / (pz - 1)) ;
	for (int i = 0; i < pz; i++) {
		u[i].re = u[i].re*factor;
		u[i].im = u[i].im*factor;
	}
	u[0].re = u[0].re*0.5;
	u[0].im = u[0].im*0.5;
}

__host__ void cheby_p2s(cudaPitchedPtr tPtr, int cmx, int my, int mz) {
	const size_t pitch = tPtr.pitch;
	const int px = cmx;
	const int py = my;
	const int pz = mz / 2 + 1;

	int threadDimx = 16;
	int threadDimy = 16;

	int blockDimx = px / threadDimx;
	int blockDimy = py / threadDimy;

	if (px%threadDimx != 0) blockDimx++;
	if (py%threadDimy != 0) blockDimy++;

	dim3 nthread(threadDimx, threadDimy);
	dim3 nBlock(blockDimx, blockDimy);

	cufftResult res;
	cudaError_t err;
	cheby_pre_p2s<<<nBlock,nthread>>>((complex*)tPtr.ptr, tPtr.pitch, cmx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
#endif
	assert(err == cudaSuccess);

	res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)tPtr.ptr,
		(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
	assert(res == CUFFT_SUCCESS);

	size_t inc = cmx*(my/3*2)* tPtr.pitch / sizeof(CUFFTCOMPLEX);

	res = CUFFTEXEC_C2C(planZ_pad, ((CUFFTCOMPLEX*)tPtr.ptr)+inc,
		((CUFFTCOMPLEX*)tPtr.ptr) + inc, CUFFT_FORWARD);
	assert(res == CUFFT_SUCCESS);

	//err = cudaDeviceSynchronize();
	//assert(err == cudaSuccess);

	cheby_post_p2s<<<nBlock, nthread>>>((complex*)tPtr.ptr, tPtr.pitch, cmx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
#endif
	assert(err == cudaSuccess);
}
__host__ void cheby_s2p(cudaPitchedPtr tPtr, int mx, int my, int mz, Padding_mode doPadding) {
	const size_t pitch = tPtr.pitch;
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;

	int threadDimx = 16;
	int threadDimy = 16;

	int blockDimx = px / threadDimx ;
	int blockDimy = py / threadDimy ;

	if (px%threadDimx != 0) blockDimx++;
	if (py%threadDimy != 0) blockDimy++;

	dim3 nthread(threadDimx, threadDimy);
	dim3 nBlock(blockDimx, blockDimy);
	cufftResult res;
	cudaError_t err;
	if(doPadding == Padding){
		cheby_pre_s2p_pad<<<nBlock, nthread >>>((complex*)tPtr.ptr, tPtr.pitch, mx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
#endif
		assert(err == cudaSuccess);

		res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)tPtr.ptr,
			(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);

		size_t inc = mx*(my / 3 * 2)* tPtr.pitch / sizeof(CUFFTCOMPLEX);
		
		res = CUFFTEXEC_C2C(planZ_pad, ((CUFFTCOMPLEX*)tPtr.ptr) + inc,
			((CUFFTCOMPLEX*)tPtr.ptr) + inc, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);

		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);
	}
	else if(doPadding == No_Padding){
		cheby_pre_s2p_noPad<<<nBlock, nthread >>>((complex*)tPtr.ptr, tPtr.pitch, mx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
#endif
		assert(err == cudaSuccess);

		res = CUFFTEXEC_C2C(planZ_no_pad, (CUFFTCOMPLEX*)tPtr.ptr,
			(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);

		size_t inc = mx*(my / 3 * 2)* tPtr.pitch / sizeof(CUFFTCOMPLEX);

		res = CUFFTEXEC_C2C(planZ_no_pad, ((CUFFTCOMPLEX*)tPtr.ptr) + inc,
			((CUFFTCOMPLEX*)tPtr.ptr) + inc, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);
	}
	else
	{
		assert(false);		
	}
}