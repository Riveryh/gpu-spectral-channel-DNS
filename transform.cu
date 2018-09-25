#include "transform.cuh"
#include <malloc.h>
#include <ASSERT.h>
#include <stdio.h>
#include "operation.h"
#include "cuRPCF.h"

cufftHandle planXYr2c, planXYc2r, planZ;




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

	//cufftPlanMany( plan *, int dim, int* n, int* inembed, int istride, int idist
	//  int* onembed, int ostride, int odist, cufftType, int batch);
	res = cufftPlanMany(&planXYr2c, 2, dim2, inembed, istride, idist,
		inembed2, istride, idist2, myCUFFT_R2C, pb.pz);
	assert(res == CUFFT_SUCCESS);
	res = cufftPlanMany(&planXYc2r, 2, dim2, inembed2, istride, idist2,
		inembed, istride, idist, myCUFFT_C2R, pb.pz);
	assert(res == CUFFT_SUCCESS);
	res = cufftPlanMany(&planZ, 1, dim1, onembed, ostride, odist,
		onembed, ostride, odist, myCUFFT_C2C, (mx/2+1)*my);
	assert(res == CUFFT_SUCCESS);
	return 0;
}

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
	int nx = (dim[0]/2+1);
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
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
	}
	else
	{
		ASSERT(dir == BACKWARD);
		err = cudaMemcpy(tbuffer, tptr, tsize, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		cudaDeviceSynchronize();
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

__host__ int transform_3d_one(DIRECTION dir, cudaPitchedPtr& Ptr,
	cudaPitchedPtr& tPtr, int* dim, int* tDim) {

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
		cuCheck( cudaMalloc3D(&(Ptr), extent),"cuMalloc");

		size_t size = Ptr.pitch*dim[1] * dim[2];
		size_t tSize = tPtr.pitch*(dim[0] / 2 + 1)*dim[1];
		buffer = (real*)malloc(size);
		tbuffer = (real*)malloc(tSize);
		ASSERT(buffer != nullptr);
		ASSERT(tbuffer != nullptr);

		//setZeros <<<1, threadDim >>> (Ptr, dim[0], dim[1], dim[2]);

#ifdef DEBUG
		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
		RPCF::write_3d_to_file("beforeREV.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
#endif //DEBUG

		//chebyshev transform in z direction
		cheby_s2p(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);

		transpose(dir, Ptr, tPtr, dim, tDim);

		res = CUFFTEXEC_C2R(planXYc2r, (CUFFTCOMPLEX*)Ptr.ptr,
			(CUFFTREAL*)Ptr.ptr);
		ASSERT(res == CUFFT_SUCCESS);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

#ifdef DEBUG
		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
		RPCF::write_3d_to_file("afterREV.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
#endif //DEBUG

		int nthreadx = 16;
		int nthready = 16;
		int nDimx = dim[1] / nthreadx;
		int nDimy = dim[2] / nthready;
		if (dim[1] % nthreadx != 0) nDimx++;
		if (dim[2] % nthready != 0) nDimy++;
		dim3 dim_num(nDimx, nDimy);
		dim3 thread_num(nthreadx, nthready);
		normalize << <dim_num, thread_num >> >
			(Ptr, dim[0], dim[1], dim[2], 1.0 / dim[0] / dim[1] );
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

#ifdef DEBUG
		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
		RPCF::write_3d_to_file("afterNORM.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
#endif //DEBUG


		err = cudaFree(tPtr.ptr);
		ASSERT(err == cudaSuccess);
		tPtr.ptr = nullptr;
	}
	else
	{
		// Ptr -> tPtr

		cuCheck(cudaMalloc3D(&(tPtr), tExtent),"cuMalloc");

		size_t size = Ptr.pitch*dim[1] * dim[2];
		size_t tSize = tPtr.pitch*(dim[0] / 2 + 1)*dim[1];
		buffer = (real*)malloc(size);
		tbuffer = (real*)malloc(tSize);
		ASSERT(buffer != nullptr);
		ASSERT(tbuffer != nullptr);

		//ASSERT(err == cudaSuccess);

#ifdef DEBUG
		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
		RPCF::write_3d_to_file("before.txt", buffer, Ptr.pitch, 2*(dim[0]/2+1), dim[1], dim[2]);
#endif //DEBUG

		ASSERT(dir == FORWARD);
		res = CUFFTEXEC_R2C(planXYr2c, (CUFFTREAL*)Ptr.ptr,
			(CUFFTCOMPLEX*)Ptr.ptr);

#ifdef DEBUG
		err = cudaMemcpy(buffer, Ptr.ptr, size, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

		RPCF::write_3d_to_file("afterXY.txt", buffer, Ptr.pitch, 2 * (dim[0] / 2 + 1), dim[1], dim[2]);
#endif // DEBUG


		ASSERT(err == cudaSuccess);

		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

		transpose(FORWARD, Ptr, tPtr, dim, tDim);

		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);

#ifdef DEBUG
		RPCF::write_3d_to_file("Transposed.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
#endif //DEBUG

		//transform in z direction
		cheby_p2s(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);

#ifdef DEBUG
		err = cudaMemcpy(tbuffer, tPtr.ptr, tSize, cudaMemcpyDeviceToHost);
		ASSERT(err == cudaSuccess);
		err = cudaDeviceSynchronize();
		ASSERT(err == cudaSuccess);
		RPCF::write_3d_to_file("afterZ.txt", tbuffer, tPtr.pitch, 2 * dim[2], (dim[0] / 2 + 1), dim[1]);
#endif //DEBUG

		//setZeros<<<1, threadDim >>>(Ptr, dim[0], dim[1], dim[2]);
		//err = cudaDeviceSynchronize();
		//ASSERT(err == cudaSuccess);

		err = cudaFree(Ptr.ptr);
		ASSERT(err == cudaSuccess);
		Ptr.ptr = nullptr;
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
		transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, indim, outdim);
		transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, indim, outdim);
		transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, indim, outdim);
		transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, indim, outdim);
		transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, indim, outdim);
		transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, indim, outdim);
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
__global__ void setZeros(cudaPitchedPtr p,int mx, int my, int mz) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int ndx = blockDim.x;
	int ndy = blockDim.y;
	size_t pitch = p.pitch; 
	real* ptr = (real*)p.ptr;
	int snz = pitch / sizeof(real);
	int nx = mx * 2 / 3;
	int ny = my * 2 / 3;
	int nz = (mz - 1) / 4 + 1;
	//e.g. nx = 256, mx = 384, the none-zero part is x = (0,127)+(256,383) 
	int zeroXstart = nx / 2;
	int zeroXend = mx - (nx / 2) - 1;
	int zeroYstart = ny / 2;
	int zeroYend = my - (ny / 2) - 1;
	//e.g. nz = 71, mz=281, none-zero part is z = (0,35)+(246,280)
	int zeroZstart = (nz + 1) / 2;
	int zeroZend = mz - zeroZstart + 1;

	for (int iz = 0; iz < mz; iz++) {
		for (int iy = (my/ndy)*idy; iy < (my / ndy)*(idy+1); iy++) {
			for (int ix = zeroXstart + (zeroXend-zeroXstart)/ndx*idx; 
				ix <= zeroXstart + (zeroXend - zeroXstart) / ndx*(idx+1); ix++) {
				ptr[iz + snz*(iy*nx + ix)] = 0;
			}
		}
	}
	for (int iz = 0; iz < mz; iz++) {
		for (int iy = zeroYstart + (zeroYend-zeroYstart)/ndy*idy; 
			iy <= zeroYstart + (zeroYend - zeroYstart) / ndy*(idy+1); iy++) {
			for (int ix = (mx/ndx)*idx; ix < (mx/ndx)*(idx+1); ix++) {
				ptr[iz + snz*(iy*nx + ix)] = 0;
			}
		}
	}
	for (int iz = zeroZstart; iz < zeroZend; iz++) {
		for (int iy = (my / ndy)*idy; iy < (my / ndy)*(idy + 1); iy++) {
			for (int ix = (mx / ndx)*idx; ix < (mx / ndx)*(idx + 1); ix++) {
				ptr[iz + snz*(iy*nx + ix)] = 0;
			}
		}
	}
}

__global__ void normalize(cudaPitchedPtr p, int mx, int my, int mz, real factor) {
	const int iy = threadIdx.x + blockIdx.x*blockDim.x;
	const int iz = threadIdx.y + blockIdx.y*blockDim.y;
	if (iy >= my || iz >= mz)return;

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
    size_t dist = pitch*(px*iy + ix)/sizeof(complex);
	u = u + dist;
	for (int i = 1; i < pz - 1; i++) {
		u[mz - i].re = u[i].re;
		u[mz - i].im = u[i].im;
	}
}

//preprocessing of chebyshev transform, spect to phy
__global__ void cheby_pre_s2p(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;
	const int nz = mz / 4;	//here, nz is the max index of z (start from 0)
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
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

__global__ void cheby_post_p2s(complex* u, const size_t pitch, const int mx, const int my, const int mz) {
	const int px = mx;
	const int py = my;
	const int pz = mz / 2 + 1;
	const int ix = threadIdx.x + blockIdx.x*blockDim.x;
	const int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix >= mx || iy >= my)return;
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
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);

	res = CUFFTEXEC_C2C(planZ, (CUFFTCOMPLEX*)tPtr.ptr,
		(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_FORWARD);
	assert(res == CUFFT_SUCCESS);
	
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);

	cheby_post_p2s<<<nBlock, nthread>>>((complex*)tPtr.ptr, tPtr.pitch, cmx, my, mz);
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);
}
__host__ void cheby_s2p(cudaPitchedPtr tPtr, int mx, int my, int mz) {
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
	cheby_pre_s2p<<<nBlock, nthread >>>((complex*)tPtr.ptr, tPtr.pitch, mx, my, mz);
	err = cudaDeviceSynchronize();
	assert(err == cudaSuccess);

	res = CUFFTEXEC_C2C(planZ, (CUFFTCOMPLEX*)tPtr.ptr,
		(CUFFTCOMPLEX*)tPtr.ptr, CUFFT_INVERSE);
	ASSERT(res == CUFFT_SUCCESS);

	err = cudaDeviceSynchronize();

	ASSERT(err == cudaSuccess);
}