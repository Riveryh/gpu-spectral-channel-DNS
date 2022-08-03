#include "../include/transform.cuh"
#include <malloc.h>
#include <assert.h>
#include <stdio.h>
#include "../include/operation.h"
#include "../include/util.h"
#include <omp.h>
#include "../include/transpose.cuh"
#include <iostream>

cufftHandle planXYr2c, planXYc2r, planZ_pad, planZ_no_pad;
cufftHandle planXYr2c_X3, planXYc2r_X6, planZ_X6, planZ_X3;

#define KERNEL_SYNCHRONIZED

cudaEvent_t start_trans, end_trans;

__host__ int initFFT(problem &pb) {
	cufftResult res;
	const int mx = pb.mx;
	const int my = pb.my;
	const int mz = pb.mz;
	const int inPitch = pb.pitch;
	const int outPitch = pb.tPitch;
	const int pmx = inPitch / sizeof(REAL);
	const int pmz = outPitch / sizeof(cuRPCF::complex);
	const int nx = mx  / 3 * 2;
	const int ny = my  / 3 * 2;

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
	if (!(res == CUFFT_SUCCESS)) {
		std::cerr << "[ERROR]:plan create failed!" << std::endl;
	};

	res = cufftPlanMany(&planXYc2r, 2, dim2, inembed2, istride, idist2,
		inembed, istride, idist, myCUFFT_C2R, pb.pz);
	if (!(res == CUFFT_SUCCESS)) {
		std::cerr << "[ERROR]:plan create failed!" << std::endl;
	};

	res = cufftPlanMany(&planZ_pad, 1, dim1, onembed, ostride, odist,
		onembed, ostride, odist, myCUFFT_C2C, (nx/2+1)*ny);
	if (!(res == CUFFT_SUCCESS)) {
		std::cerr << "[ERROR]:plan create failed!" << std::endl;
	};

	res = cufftPlanMany(&planZ_no_pad, 1, dim1_no_pad, onembed, ostride, odist,
		onembed, ostride, odist, myCUFFT_C2C, (nx/2+1)*ny);
	if (!(res == CUFFT_SUCCESS)) {
		std::cerr << "[ERROR]:plan create failed !" << std::endl;
	};

	//res = cufftPlanMany(&planXYr2c_X3, 2, dim2, inembed, istride, idist,
	//	inembed2, istride, idist2, myCUFFT_R2C, pb.pz*3);
	//assert(res == CUFFT_SUCCESS);
	//res = cufftPlanMany(&planXYc2r_X6, 2, dim2, inembed2, istride, idist2,
	//	inembed, istride, idist, myCUFFT_C2R, pb.pz*6);

	//res = cufftPlanMany(&planZ_X3, 1, dim1, onembed, ostride, odist,
	//	onembed, ostride, odist, myCUFFT_C2C, (nx / 2 + 1)*ny*3);
	//res = cufftPlanMany(&planZ_X6, 1, dim1, onembed, ostride, odist,
	//	onembed, ostride, odist, myCUFFT_C2C, (nx / 2 + 1)*ny * 6);


	assert(res == CUFFT_SUCCESS);

	assert(res == CUFFT_SUCCESS);

	cudaEventCreate(&start_trans);
	cudaEventCreate(&end_trans);

	return 0;
}

__host__ int transform_3d_one(DIRECTION dir, cudaPitchedPtr& Ptr,
	cudaPitchedPtr& tPtr, int* dim, int* tDim, 
	Padding_mode pd, bool isOutput) {

	//transform in x-y direction
	cufftResult res;

	cudaExtent extent = make_cudaExtent(
	  2*(dim[0]/2+1) * sizeof(REAL), dim[1], dim[2]);
	cudaError_t err;

	ASSERT(dim[0] == tDim[1]);
	ASSERT(dim[1] == tDim[2]);
	ASSERT(dim[2] == tDim[0]);

	int nx = dim[0]  / 3 * 2;
	int ny = dim[1]  / 3 * 2;

	cudaExtent tExtent = make_cudaExtent(
		tDim[0] * sizeof(cuRPCF::complex), nx/2+1 , ny);
	cudaExtent pExtent = make_cudaExtent(
		2 * (dim[0] / 2 + 1) * sizeof(REAL), dim[1], dim[2]/2+1);

	dim3 threadDim(4, 4);

//	REAL* buffer;
//	REAL* tbuffer;
	float time;
	// tPtr -> Ptr
	if (dir == BACKWARD) {

//		size_t size = Ptr.pitch*dim[1] * dim[2];
//		size_t pSize = Ptr.pitch*dim[1] * (dim[2]/2+1);
//		size_t tSize = tPtr.pitch*(nx / 2 + 1)*ny;
//		buffer = (REAL*)malloc(size);
//		tbuffer = (REAL*)malloc(tSize);
//		ASSERT(buffer != nullptr);
//		ASSERT(tbuffer != nullptr);

		//setZeros <<<1, threadDim >>> (Ptr, dim[0], dim[1], dim[2]);

//#ifdef DEBUG
//		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if(isOutput) RPCF::write_3d_to_file("beforeREV.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
//#endif //DEBUG
		//chebyshev transform in z direction
			cheby_s2p(tPtr, dim[0] / 2 + 1, dim[1] , dim[2], pd);

		//transpose(dir, Ptr, tPtr, dim, tDim);
#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(start_trans);
#endif 

		CUDA_CHECK(myCudaMalloc(Ptr, XYZ_3D));
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);
		CUDA_CHECK(myCudaFree(tPtr, ZXY_3D));

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "transpose backward time = " << time / 1000.0 << std::endl;
		cudaEventRecord(start_trans);

#endif
		
			setZeros((cuRPCF::complex*)Ptr.ptr, Ptr.pitch, dim3(dim[0], dim[1], dim[2]));

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "set zeros time = " << time / 1000.0 << std::endl;

		cudaEventRecord(start_trans);
#endif 

			void* dev_buffer = get_fft_buffer_ptr();
			res = CUFFTEXEC_C2R(planXYc2r, (CUFFTCOMPLEX*)Ptr.ptr,
				(CUFFTREAL*)Ptr.ptr);
		
			//(CUFFTREAL*)dev_buffer);
		//cuCheck(cudaMemcpy(Ptr.ptr, dev_buffer, pSize, cudaMemcpyDeviceToDevice),"mem move");

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "FFT XY BACKWARD TIME = " << time / 1000.0 << std::endl;
#endif

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

	}
	else
	{
		// Ptr -> tPtr

//		size_t size = Ptr.pitch*dim[1] * dim[2];
//		size_t pSize = Ptr.pitch*dim[1] * (dim[2] / 2 + 1);
//		size_t tSize = tPtr.pitch*(dim[0] / 2 + 1)*dim[1];
//		buffer = (REAL*)malloc(size);
//		tbuffer = (REAL*)malloc(tSize);
//		ASSERT(buffer != nullptr);
//		ASSERT(tbuffer != nullptr);

		//ASSERT(err == cudaSuccess);

//#ifdef DEBUG
//		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
//		ASSERT(err == cudaSuccess);
//		err = cudaDeviceSynchronize();
//		ASSERT(err == cudaSuccess);
//		if (isOutput) RPCF::write_3d_to_file("before.txt", buffer, Ptr.pitch, 2*(dim[0]/2+1), dim[1], dim[2]);
//#endif //DEBUG

		ASSERT(dir == FORWARD);
		void* dev_buffer = get_fft_buffer_ptr();

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(start_trans);
#endif
		
		res = CUFFTEXEC_R2C(planXYr2c, (CUFFTREAL*)Ptr.ptr,
			(CUFFTCOMPLEX*)Ptr.ptr); 

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)	
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "FFT XY forward TIME = " << time / 1000.0 << std::endl;
#endif
			//(CUFFTCOMPLEX*)dev_buffer);
		//cuCheck(cudaMemcpy(Ptr.ptr, dev_buffer, pSize, cudaMemcpyDeviceToDevice), "mem move");
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

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(start_trans);
#endif 
		
		normalize(Ptr, dim3(dim[0], dim[1], dim[2]), 1.0 / dim[0] / dim[1]);

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "normalize TIME = " << time / 1000.0 << std::endl;

		//transpose(FORWARD, Ptr, tPtr, dim, tDim);
		cudaEventRecord(start_trans);
#endif
		CUDA_CHECK(myCudaMalloc(tPtr, ZXY_3D));
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);
		CUDA_CHECK(myCudaFree(Ptr, XYZ_3D));

		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "tranpose forward TIME = " << time / 1000.0 << std::endl;
#endif
		//err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
		//ASSERT(err == cudaSuccess);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);

//#ifdef DEBUG
//		if (isOutput) RPCF::write_3d_to_file("Transposed.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
//#endif //DEBUG

		//transform in z direction
		cheby_p2s(tPtr, dim[0] / 2 + 1, dim[1], dim[2], pd);

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
		
	}
//	free(buffer);
//	free(tbuffer);
	return 0;
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
		//transform_backward_X6(pb);
		//return 0;
		transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, indim, outdim, Padding);
		transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, indim, outdim, Padding);
	}
	if (dir == FORWARD) {
		//transform_forward_X3(pb);
		//return 0;
		transform_3d_one(FORWARD, pb.dptr_lamb_z, pb.dptr_tLamb_z, indim, outdim);
		transform_3d_one(FORWARD, pb.dptr_lamb_y, pb.dptr_tLamb_y, indim, outdim);
		transform_3d_one(FORWARD, pb.dptr_lamb_x, pb.dptr_tLamb_x, indim, outdim);
	}
	return 0;
}

//mx, my, mz is the size of large matrix
//nx, ny, nz is the size of the small matrix (dealiased)
__global__ void setZerosKernel(cuRPCF::complex* ptr,size_t pitch, int mx, int my, int mz) {
	int ky =  blockIdx.x;
	int kz =  blockIdx.y;
	int kx = threadIdx.x;
	if (ky >= my || kz >= mz/2+1 || kx>= mx/2+1) return;
	assert(kx * sizeof(cuRPCF::complex) <= pitch);
	size_t inc = pitch * (kz * my + ky)/sizeof(cuRPCF::complex);
	ptr = ptr + inc;
	int nx = mx / 3 * 2;
	int ny = my / 3 * 2;
	
	if (ky >= ny / 2 && ky <= my - (ny/2)) {
		ptr[kx] = 0.0;
		return;
	}
	else
	{
		if( kx >= nx/2 ) {
			ptr[kx] = 0.0;
		}
		return;
	}
}

__host__ void setZeros(cuRPCF::complex* ptr, size_t pitch, dim3 dims) {
	int dim[3] = { dims.x,dims.y,dims.z };
	setZerosKernel <<<dim3(dims.y,dims.z/2+1), dims.x/2+1 >>>((cuRPCF::complex*)ptr, pitch,
		dim[0], dim[1], dim[2]);
//#ifdef KERNEL_SYNCHRONIZED
	CUDA_CHECK(cudaDeviceSynchronize());
//#endif
}

__global__ void normalizeKernel(REAL* ptr, size_t pitch , int mx, int my, int mz, REAL factor) {
	const int iy = blockIdx.x;
	const int iz = blockIdx.y;
	const int ix = threadIdx.x;
	//if (iy >= my || iz >= mz/2+1)return;
	//const int ny = my / 3 * 2;
	//if (iy > ny / 2  && iy < my - (ny/2)) return;
	//if (ix >= mx) return;

	size_t dist = pitch*(my*iz + iy) / sizeof(cuRPCF::complex);

	cuRPCF::complex* row = ((cuRPCF::complex*)ptr) + dist;
	row[ix] = row[ix] * factor;
}

__host__ void normalize(cudaPitchedPtr Ptr, dim3 dims, REAL factor) {
	cudaError_t err;
	int dim[3] = { dims.x,dims.y,dims.z }; 
	dim3 nDim(dim[1], dim[2] / 2 + 1);
	normalizeKernel<<<nDim, dim[0]/2+1>>> ((REAL*)Ptr.ptr, Ptr.pitch, dim[0], dim[1], dim[2], factor);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
#endif
	ASSERT(err == cudaSuccess);
}


//preprocessing of chebyshev transform, spect to phy
__global__ void cheby_pre_s2p_pad(cuRPCF::complex* u, const size_t pitch, const int hmx, const int my, const int mz) {
	const int mx = (hmx-1)*2;
	const int pz = mz / 2 + 1;
	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int hnx = mx / 3 * 2 / 2 + 1;
	const int ny = my / 3 * 2;
	const int ix = blockIdx.x;
	const int iy = blockIdx.y;
	if (ix >= hnx || iy >= ny)return;
	const int iz = threadIdx.x;
	if (iz > nz)return;

	size_t dist = pitch*(hnx*iy + ix) / sizeof(cuRPCF::complex);
	u = u + dist;
	/*for (int i = nz; i < pz; i++) {
		u[i].re = 0.0;
		u[i].im = 0.0;
	}*/
	u[iz + nz + 1] = 0.0;
	u[iz + pz - 1] = 0.0;
	/*for (int i = 0; i < nz; i++) {
		u[i].re = u[i].re*0.5;
		u[i].im = u[i].im*0.5;
	}*/
	u[iz] = u[iz] * 0.5;

	/*for (int i = 1; i < pz - 1; i++) {
		u[mz - i].re = u[i].re;
		u[mz - i].im = u[i].im;
	}*/

	if (iz == 0) {
		u[0] = u[0] * 2.0;
	}
	else {
		u[mz - iz] = u[iz];
	}
}

__global__ void cheby_pre_s2p_noPad(cuRPCF::complex* u, const size_t pitch, const int hmx, const int my, const int mz) {
	const int mx = (hmx - 1) * 2;
	const int pz = mz / 2 + 1;
	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int hnx = mx/ 3 * 2 / 2 + 1;
	const int ny = my / 3 * 2;
	const int ix = blockIdx.x;
	const int iy = blockIdx.y;
	const int iz = threadIdx.x;
	if (ix >= hnx || iy >= ny)return;

	size_t dist = pitch*(hnx*iy + ix) / sizeof(cuRPCF::complex);

	u = u + dist;
	//for (int i = nz; i < pz; i++) {
	//	u[i].re = 0.0;
	//	u[i].im = 0.0;
	//}
	int i = iz;
	//for (int i = 0; i < nz; i++) {
	if (i <= nz) {
		u[i].re = u[i].re*0.5;
		u[i].im = u[i].im*0.5;
	}

	__syncthreads();
	//for (int i = 1; i < nz - 1; i++) {

	if (i >= 1 && i <= nz - 1) {
		u[pz - 1 - i].re = u[i].re;
		u[pz - 1 - i].im = u[i].im;
	}else if (i == 0 || i==nz) {
	//}else if (i == 0) {
		u[i].re = u[i].re*2.0;
		u[i].im = u[i].im*2.0;
	}
}

//preprocessing of chebyshev transform, physical to spectral
__global__ void cheby_pre_p2s(cuRPCF::complex* u, const size_t pitch, const int hmx, const int my, const int mz) {
	const int mx = (hmx - 1) * 2;
	const int pz = mz / 2 + 1;
	//	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int hnx = mx / 3 * 2 / 2 + 1;
	const int ny = my / 3 * 2;
	const int ix = blockIdx.x;
	const int iy = blockIdx.y;
	const int iz = threadIdx.x;
	if (ix >= hnx || iy >= ny)return;
	if (iz >= pz - 1) return;
	if (iz == 0)return;
	size_t dist = pitch*(hnx*iy + ix) / sizeof(cuRPCF::complex);
	u = u + dist;
	u[mz - iz].re = u[iz].re;
	u[mz - iz].im = u[iz].im;
}

//post-processing of chebyshev transform, physical to spectral
__global__ void cheby_post_p2s(cuRPCF::complex* u, const size_t pitch, const int hmx, const int my, const int mz) {
	const int mx = (hmx - 1) * 2;
	const int pz = mz / 2 + 1;
	//const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int hnx = mx/ 3 * 2 /2 + 1;
	const int ny = my / 3 * 2;
	const int ix = blockIdx.x;
	const int iy = blockIdx.y;
	if (ix >= hnx || iy >= ny)return;
	const int iz = threadIdx.x;
	if (iz >= pz)return;
	size_t dist = pitch*(hnx*iy + ix) / sizeof(cuRPCF::complex);

	u = u + dist;
	REAL factor = (1.0 / (pz - 1));

	u[iz].re = u[iz].re*factor;
	u[iz].im = u[iz].im*factor;

	if (iz == 0 || iz == pz-1) {
		u[iz].re = u[iz].re*0.5;
		u[iz].im = u[iz].im*0.5;
	}
}

__host__ void cheby_p2s(cudaPitchedPtr tPtr, int hmx, int my, int mz, Padding_mode pad) {
//	const size_t pitch = tPtr.pitch;
	const int nx = (hmx - 1) * 2 / 3 * 2;
	const int ny = my / 3 * 2;
	const int hnx = nx / 2 + 1;

	//int threadDimx = 16;
	//int threadDimy = 16;

	//int blockDimx = hnx / threadDimx;
	//int blockDimy = ny / threadDimy;

	//if (hnx%threadDimx != 0) blockDimx++;
	//if (ny%threadDimy != 0) blockDimy++;

	//dim3 nthread(threadDimx, threadDimy);
	//dim3 nBlock(blockDimx, blockDimy);

	cufftResult res;
	cudaError_t err;
	float time;
#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
	cudaEventRecord(start_trans);
#endif


	// Transform with dealiasing
	if (pad == Padding) {
		cheby_pre_p2s <<< dim3(hnx, ny), mz / 2 + 1 >> > ((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
		assert(err == cudaSuccess);
#endif

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby_pre_p2s_time = " << time / 1000.0 << std::endl;

		cudaEventRecord(start_trans);
#endif 

		res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)tPtr.ptr,
			(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
		assert(res == CUFFT_SUCCESS);

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby fft p2s time = " << time / 1000.0 << std::endl;

		//err = cudaDeviceSynchronize();
		//assert(err == cudaSuccess);

		cudaEventRecord(start_trans);
#endif

		cheby_post_p2s << <dim3(hnx, ny), mz / 2 + 1 >> > ((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
		assert(err == cudaSuccess);
#endif

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby_post_p2s_time = " << time / 1000.0 << std::endl;
#endif
	}
	else //Transform without dealiasing
	{
		{
			cheby_pre_p2s << <dim3(hnx, ny), mz / 4 + 1 >> > ((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz/2);
#ifdef KERNEL_SYNCHRONIZED
			err = cudaDeviceSynchronize();
			assert(err == cudaSuccess);
#endif

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
			cudaEventRecord(end_trans);
			cudaEventSynchronize(end_trans);
			cudaEventElapsedTime(&time, start_trans, end_trans);
			std::cout << "cheby_pre_p2s_time = " << time / 1000.0 << std::endl;

			cudaEventRecord(start_trans);
#endif

			res = CUFFTEXEC_C2C(planZ_no_pad, (CUFFTCOMPLEX*)tPtr.ptr,
				(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
			assert(res == CUFFT_SUCCESS);

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
			cudaEventRecord(end_trans);
			cudaEventSynchronize(end_trans);
			cudaEventElapsedTime(&time, start_trans, end_trans);
			std::cout << "cheby fft p2s time = " << time / 1000.0 << std::endl;

			//err = cudaDeviceSynchronize();
			//assert(err == cudaSuccess);

			cudaEventRecord(start_trans);
#endif

			cheby_post_p2s <<<dim3(hnx, ny), mz / 4 + 1 >> > ((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz/2);
#ifdef KERNEL_SYNCHRONIZED
			err = cudaDeviceSynchronize();
			assert(err == cudaSuccess);
#endif

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
			cudaEventRecord(end_trans);
			cudaEventSynchronize(end_trans);
			cudaEventElapsedTime(&time, start_trans, end_trans);
			std::cout << "cheby_post_p2s_time = " << time / 1000.0 << std::endl;
#endif
		}
	}
}

//spectral to physical chebyshev transform in wall-normall direction
__host__ void cheby_s2p(cudaPitchedPtr tPtr, int hmx, int my, int mz, Padding_mode doPadding) {
//	const size_t pitch = tPtr.pitch;
//	const int pz = mz / 2 + 1;
	const int nx = (hmx-1)*2/3*2;
	const int ny = my/3*2;
	const int hnx = nx/2+1;


	cufftResult res;
	cudaError_t err;
	float time;

	if(doPadding == Padding){

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(start_trans);
#endif

		cheby_pre_s2p_pad<<<dim3(hnx,ny), mz/4+1 >>>((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz);

#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
		assert(err == cudaSuccess);
#endif	

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby_pre_s2p_pad_time = " << time / 1000.0 << std::endl;


		cudaEventRecord(start_trans);
#endif

		res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)tPtr.ptr,
			(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);

#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)		
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby fft s2p padding time = " << time / 1000.0 << std::endl;
#endif


		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);
	}
	else if(doPadding == No_Padding)
	{
#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(start_trans);
#endif
		cheby_pre_s2p_noPad<<<dim3(hnx,ny), mz/4 >>>((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my, mz);
#ifdef KERNEL_SYNCHRONIZED
		err = cudaDeviceSynchronize();
		assert(err == cudaSuccess);
#endif
#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby_pre_s2p_nopad_time = " << time / 1000.0 << std::endl;


		cudaEventRecord(start_trans);
#endif

		res = CUFFTEXEC_C2C(planZ_no_pad, (CUFFTCOMPLEX*)tPtr.ptr,
			(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
		ASSERT(res == CUFFT_SUCCESS);
		
#if (defined CURPCF_CUDA_PROFILING) && (defined SHOW_TRANSFORM_TIME)
		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);
		std::cout << "cheby fft s2p no pad time = " << time / 1000.0 << std::endl;
#endif
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);
	}
	else
	{
		assert(false);		
	}
}

__host__ void transform_backward_X6(problem& pb) {
	int dim[3] = { pb.mx,pb.my,pb.mz };
	int tDim[3] = { pb.mz,pb.mx,pb.my };

	cheby_s2p_X6(pb.dptr_tu, dim[0] / 2 + 1, dim[1], dim[2]);

	//transpose(dir, Ptr, tPtr, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	cuda_transpose(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);

	cudaPitchedPtr& Ptr = pb.dptr_u;

	int nThreadx = 16;
	int nThready = 16;
	dim3 nThread(nThreadx, nThready);
	int nDimx = dim[1] / nThreadx;
	int nDimy = (dim[2] / 2 + 1)*6 / nThready;
	if (dim[1] % nThreadx != 0) nDimx++;
	if ((dim[2] / 2 + 1)*6 % nThready != 0) nDimy++;
	dim3 nDim(nDimx, nDimy);
	setZerosKernel<<<nDim, nThread >>>((cuRPCF::complex*)Ptr.ptr, Ptr.pitch,
		dim[0], dim[1], dim[2]*6);
#ifdef KERNEL_SYNCHRONIZED
	CUDA_CHECK(cudaDeviceSynchronize());
#endif
	cufftResult_t res;
	res = CUFFTEXEC_C2R(planXYc2r_X6, (CUFFTCOMPLEX*)pb.dptr_u.ptr,
		(CUFFTREAL*)pb.dptr_u.ptr);
	ASSERT(res == CUFFT_SUCCESS);
	CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void transform_forward_X3(problem& pb) {
	cufftResult_t res;
	cudaPitchedPtr Ptr = pb.dptr_lamb_x;
	res = CUFFTEXEC_R2C(planXYr2c_X3, (CUFFTREAL*)Ptr.ptr,
		(CUFFTCOMPLEX*)Ptr.ptr);
	
	int dim[3] = { pb.mx, pb.my, pb.mz };
	int tDim[3] = { pb.mz, pb.mx, pb.my };
	//normalize;
	int nthreadx = 16;
	int nthready = 16;
	int nDimx = dim[1] / nthreadx;
	int nDimy = (dim[2] / 2 + 1) * 3/ nthready;
	if (dim[1] % nthreadx != 0) nDimx++;
	if ((dim[2] / 2 + 1)*3 % nthready != 0) nDimy++;
	dim3 dim_num(nDimx, nDimy);
	dim3 thread_num(nthreadx, nthready);

	// THIS LAUNCH PARAMETER NEED TO BE CHANGED
	normalizeKernel<<< dim_num, thread_num >>>
		((REAL*)Ptr.ptr, Ptr.pitch, dim[0], dim[1], dim[2]*3, 1.0 / dim[0] / dim[1]);
	CUDA_CHECK(cudaDeviceSynchronize());


	cuda_transpose(FORWARD, pb.dptr_lamb_z, pb.dptr_tLamb_z, dim, tDim);
	cuda_transpose(FORWARD, pb.dptr_lamb_y, pb.dptr_tLamb_y, dim, tDim);
	cuda_transpose(FORWARD, pb.dptr_lamb_x, pb.dptr_tLamb_x, dim, tDim);
	
	cheby_p2s_X3(pb.dptr_tLamb_x, dim[0] / 2 + 1, dim[1], dim[2]);
}

__host__ void cheby_p2s_X3(cudaPitchedPtr tPtr, int hmx, int my, int mz) {
//	const size_t pitch = tPtr.pitch;
	const int nx = (hmx - 1) * 2 / 3 * 2;
	const int ny = my / 3 * 2;
	const int hnx = nx / 2 + 1;

	int threadDimx = 16;
	int threadDimy = 16;

	int blockDimx = hnx / threadDimx;
	int blockDimy = ny*3 / threadDimy;

	if (hnx%threadDimx != 0) blockDimx++;
	if (ny*3%threadDimy != 0) blockDimy++;

	dim3 nthread(threadDimx, threadDimy);
	dim3 nBlock(blockDimx, blockDimy);

	cufftResult res;
	cudaError_t err;
	cheby_pre_p2s <<<nBlock, nthread >> >((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my*3, mz);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);
#endif

	res = CUFFTEXEC_C2C(planZ_X3, (CUFFTCOMPLEX*)tPtr.ptr,
		(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
	assert(res == CUFFT_SUCCESS);

	//err = cudaDeviceSynchronize();
	//assert(err == cudaSuccess);

	cheby_post_p2s <<<nBlock, nthread >>>((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, my*3, mz);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);
#endif
}


__host__ void cheby_s2p_X6(cudaPitchedPtr tPtr, int hmx, int my, int mz) {
//	const size_t pitch = tPtr.pitch;
	//const int pz = mz / 2 + 1;
	const int nx = (hmx - 1) * 2 / 3 * 2;
	const int ny = my / 3 * 2;
	const int hnx = nx / 2 + 1;

	int threadDimx = 16;
	int threadDimy = 16;

	int blockDimx = hnx / threadDimx;
	int blockDimy = 6*ny / threadDimy;

	if (hnx%threadDimx != 0) blockDimx++;
	if (6*ny%threadDimy != 0) blockDimy++;

	dim3 nthread(threadDimx, threadDimy);
	dim3 nBlock(blockDimx, blockDimy);
	cufftResult res;
	cudaError_t err;
	cheby_pre_s2p_pad <<<nBlock, nthread >>>((cuRPCF::complex*)tPtr.ptr, tPtr.pitch, hmx, 6*my, mz);
#ifdef KERNEL_SYNCHRONIZED
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);
#endif		

	res = CUFFTEXEC_C2C(planZ_X6, (CUFFTCOMPLEX*)tPtr.ptr,
		(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
	ASSERT(res == CUFFT_SUCCESS);

	//err = cudaDeviceSynchronize();
	//ASSERT(err == cudaSuccess);	
}