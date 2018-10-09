#include "transpose.cuh"
#include <cassert>
#include <iostream>
#include "cuRPCF.h"

__global__ void transpose_forward(complex* u, complex* tu, dim3 dim,
	size_t pitch, size_t tPitch);

__global__ void transpose_backward(complex* u, complex* tu, dim3 dim,
	size_t pitch, size_t tPitch);

// dim参数表示的是xyz排列的数据的维度（Real格式），实际上由于transpose时存储的是Complex格式，
// 因此实际的数据维数为 2*(dim[0]/2+1) x dim[1] x dim[2]
// tDim参数表示的是zxy排列的数据的维度（Real格式），实际上由于transpose时存储的是Complex格式，
// 因此实际的数据维数为 dim[2] x 2*(dim[0]/2+1) x dim[1] 
// 或 tDim[0] x (tDim[1]/2+1)*2 x tDim[2]
// 实际上tDim并不会使用，仅供程序验证
__host__ int transpose(DIRECTION dir, cudaPitchedPtr Ptr,
	cudaPitchedPtr tPtr, int* dim, int* tDim) {
	//storage of host temporal variable
	complex* buffer, *tbuffer;

	//number of complex
	int nx = (dim[0] / 2 + 1);
	int ny = dim[1];
	int nz = dim[2];
	size_t Pitch = Ptr.pitch;
	size_t tPitch = tPtr.pitch;

	// sizes are defined in unit of bytes
	size_t size = Pitch * ny * nz;
	size_t tsize = tPitch * nx * ny;

	complex* ptr = (complex*)Ptr.ptr;
	complex* tptr = (complex*)tPtr.ptr;
	cudaError_t err;

	//ASSERT(dir == BACKWARD);
	ASSERT(sizeof(complex) == 2 * sizeof(real));
	ASSERT(dim[0] == tDim[1]);
	ASSERT(dim[1] == tDim[2]);
	ASSERT(dim[2] == tDim[0]);
	ASSERT(Pitch >= nx * sizeof(complex));
	ASSERT(tPitch >= nz * sizeof(complex));
	ASSERT(ptr != nullptr);
	ASSERT(tptr != nullptr);

	buffer = (complex*)malloc(size);
	tbuffer = (complex*)malloc(tsize);

	//set default value to zeros.
	for (size_t i = 0; i < size / sizeof(complex); i++) {
		buffer[i] = 0.0;
	}
	for (size_t i = 0; i < tsize / sizeof(complex); i++) {
		tbuffer[i] = 0.0;
	}

	size_t layerIn = Pitch / sizeof(complex)*ny;
	// Ptr[z][y][x] = tPtr[y][x][z]
	if (dir == FORWARD) {
		err = cudaMemcpy(buffer, ptr, size, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
#pragma omp parallel for
		for (int k = 0; k < nz; k++) {
			for (int j = 0; j < ny; j++) {
				for (int i = 0; i < nx; i++) {
					size_t index1 = k*layerIn + Pitch / sizeof(complex)*j + i;
					size_t index2 = (nx*j + i)*tPitch / sizeof(complex) + k;
					tbuffer[index2] = buffer[index1];
				}
			}
		}
		err = cudaMemcpy(tptr, tbuffer, tsize, cudaMemcpyHostToDevice);
		ASSERT(err == cudaSuccess);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);
	}
	else
	{
		ASSERT(dir == BACKWARD);
		err = cudaMemcpy(tbuffer, tptr, tsize, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		//cudaDeviceSynchronize();
#pragma omp parallel for
		for (int k = 0; k < nz; k++) {
			for (int j = 0; j < ny; j++) {
				for (int i = 0; i < nx; i++) {
					size_t index1 = k*layerIn + Pitch / sizeof(complex)*j + i;
					size_t index2 = (nx*j + i)*tPitch / sizeof(complex) + k;
					buffer[index1] = tbuffer[index2];
				}
			}
		}
		err = cudaMemcpy(ptr, buffer, size, cudaMemcpyHostToDevice);
		ASSERT(cudaSuccess == err);
	}
	//RPCF::write_3d_to_file("input.txt", (real*)buffer, Ptr.pitch,
	//	nx, ny, nz);
	//RPCF::write_3d_to_file("output.txt", (real*)tbuffer, tPtr.pitch,
	//	nz, nx, ny);
	//investigate << <1, 1 >> > (Ptr);
	//investigate << <1, 1 >> > (tPtr);
	free(buffer);
	free(tbuffer);
	return 0;
}

__host__ int cuda_transpose(DIRECTION dir, cudaPitchedPtr Ptr,
	cudaPitchedPtr tPtr, int* dim, int* tDim) {
	int nthreadx = 16;
	int nthready = 16;
	
	dim3 dims(dim[0], dim[1], dim[2]);
	if (dir == FORWARD) {
		int nBlockx = dim[0] / nthreadx;
		int nBlocky = dim[1] / nthready;
		if (dim[0] % nthreadx != 0) nBlockx++;
		if (dim[1] % nthready != 0) nBlocky++;
		dim3 nBlock(nBlockx, nBlocky);
		dim3 nThread(nthreadx, nthready);

		transpose_forward<<<nBlock,nThread>>>((complex*)Ptr.ptr, (complex*)tPtr.ptr,
			dims, Ptr.pitch, tPtr.pitch);
		//DEBUG:cuCheck(cudaDeviceSynchronize(),"Transpose kernel");
	}
	else if (dir == BACKWARD) {
		int nBlockx = dim[1] / nthreadx;
		int nBlocky = dim[2]/2+1 / nthready;
		if (dim[1] % nthreadx != 0) nBlockx++;
		if (dim[2]/2+1 % nthready != 0) nBlocky++;
		dim3 nBlock(nBlockx, nBlocky);
		dim3 nThread(nthreadx, nthready);

		transpose_backward<<<nBlock,nThread>>>((complex*)Ptr.ptr, (complex*)tPtr.ptr,
			dims, Ptr.pitch, tPtr.pitch);
		//DEBUG:cuCheck(cudaDeviceSynchronize(), "Transpose kernel");
	}
	else {
		std::cerr << "Wrong tranpose type!" << std::endl;
	}
	return 0;
}

__global__ void transpose_forward(complex* u, complex* tu, dim3 dim,
	size_t pitch, size_t tPitch) {
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	int ky = threadIdx.y + blockDim.y*blockIdx.y;

	int mx = dim.x;
	int my = dim.y;
	int mz = dim.z;
	if (kx >= mx / 2 + 1) return;
	if (ky >= my) return;
	for (int kz = 0; kz < mz/2+1; kz++) {
		size_t inc = pitch / sizeof(complex)*(kz*my + ky) + kx;
		size_t tInc = tPitch / sizeof(complex)*(ky*((mx / 2) + 1) + kx) + kz;
		tu[tInc] = u[inc];
	}
}

__global__ void transpose_backward(complex* u, complex* tu, dim3 dim,
	size_t pitch, size_t tPitch) {
	int ky = threadIdx.x + blockDim.x*blockIdx.x;
	int kz = threadIdx.y + blockDim.y*blockIdx.y;

	int mx = dim.x;
	int my = dim.y;
	int mz = dim.z;
	if (kz >= mz / 2 + 1) return;
	if (ky >= my) return;
	for (int kx = 0; kx < mx/2+1; kx++) {
		size_t inc = pitch / sizeof(complex)*(kz*my + ky) + kx;
		size_t tInc = tPitch / sizeof(complex)*(ky*((mx / 2) + 1) + kx) + kz;
		u[inc] = tu[tInc];
	}
}