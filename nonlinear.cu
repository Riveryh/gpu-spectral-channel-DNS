#include "nonlinear.cuh"
#include "assert.h"
#include "cuRPCF.h"
#include <iostream>

__host__ int addMeanFlow(problem& pb);

__host__ int addCoriolisForce(problem& pb);

__device__ void computeLambDevice(real* pU, real* pV, real* pW,
	real* pOmegaX, real* pOmegaY, real* pOmegaZ,
	real* pLambX, real* pLambY, real* pLambZ, int mz, real Ro);

__global__ void addMeanFlowKernel(cudaPitchedPtr ptr, int px, int py, int pz);
__global__ void computeLambVectorKernel(cudaPitchedPtrList pList, 
	int mx, int my, int mz, real Ro);

__host__ void saveZeroWaveLamb(problem& pb);



cudaEvent_t __start, __stop;
bool cudaTimeInitialized = false;
// Get the nonliear part of RHS
__host__ int getNonlinear(problem& pb) {
	float time;

	if (!cudaTimeInitialized) {
		cudaEventCreate(&__start);
		cudaEventCreate(&__stop);
		cudaTimeInitialized = true;
	}

	
	// spec --> phys, tPtr(z,x,y) --> Ptr(x,y,z)
	//transform(BACKWARD, pb);
	cudaEventRecord(__start, 0); 

	transform(BACKWARD, pb);

	cudaEventRecord(__stop, 0);
	cudaEventSynchronize(__stop);
	cudaEventElapsedTime(&time, __start, __stop);
	std::cout << "tranform backward time = " << time / 1000.0 << std::endl;


	// use Ptr in pb
	addMeanFlow(pb);

	cudaEventRecord(__start, 0);
	
	computeLambVector(pb);

	cudaEventRecord(__stop, 0);
	cudaEventSynchronize(__stop);
	cudaEventElapsedTime(&time, __start, __stop);
	std::cout << "get lamb time = " << time/1000.0 << std::endl;

	// phys --> spec, Ptr(x,y,z) --> tPtr(z,x,y)
	// the forward transform only deals with lamb vectors !
	// see the implementation of the function for further details.

	cudaEventRecord(__start, 0);

	transform(FORWARD, pb);


	cudaEventRecord(__stop, 0);
	cudaEventSynchronize(__stop);
	cudaEventElapsedTime(&time, __start, __stop);
	std::cout << "transform forward time = " << time/1000.0 << std::endl;

	addCoriolisForce(pb);

	// this operation will be finished in get_rhs_v;
	//saveZeroWaveLamb(pb);

	// Get the right hand side (RHS) part of the equation.
	// transform the nonlinear RHS part into phys space
	// transform(BACKWARD, pb);
	// the transfomr is done inside the function.
	cudaEventRecord(__start, 0);

	rhsNonlinear(pb);

	cudaEventRecord(__stop, 0);
	cudaEventSynchronize(__stop);
	cudaEventElapsedTime(&time, __start, __stop);
	std::cout << "get rhs non time = " << time / 1000.0 << std::endl;
	return 0;
}

__host__ int addMeanFlow(problem & pb)
{
	cudaError_t err;

	//int nthreadx = 16;
	//int nthready = 16;
	//int nDimx = pb.my / nthreadx;
	//int nDimy = pb.mz / nthready;
	//if (pb.my % nthreadx != 0) nDimx++;
	//if (pb.mz % nthready != 0) nDimy++;
	//dim3 nThread(nthreadx, nthready);
	//dim3 nDim(nDimx, nDimy);
	addMeanFlowKernel <<<pb.nDim, pb.nThread>>>(pb.dptr_u, pb.px, pb.py, pb.pz);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);
	return int();
}

__host__ int computeLambVector(problem & pb)
{
	//cudaExtent& pExtent = pb.pExtent;
	//make_cudaExtent(
	//	2 * (pb.mx / 2 + 1) * sizeof(real), pb.my, pb.mz);

	ASSERT(pb.dptr_lamb_x.ptr == nullptr);
	ASSERT(pb.dptr_lamb_y.ptr == nullptr);
	ASSERT(pb.dptr_lamb_z.ptr == nullptr);
	pb.dptr_lamb_x = pb.dptr_u;
	pb.dptr_lamb_y = pb.dptr_v;
	pb.dptr_lamb_z = pb.dptr_w;
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_x), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_y), pExtent), "allocate");
	//cuCheck(cudaMalloc3D(&(pb.dptr_lamb_z), pExtent), "allocate");

	cudaPitchedPtrList pList;
	pList.dptr_u = pb.dptr_u;
	pList.dptr_v = pb.dptr_v;
	pList.dptr_w = pb.dptr_w;
	pList.dptr_omega_x = pb.dptr_omega_x;
	pList.dptr_omega_y = pb.dptr_omega_y;
	pList.dptr_omega_z = pb.dptr_omega_z;
	pList.dptr_lamb_x = pb.dptr_lamb_x;
	pList.dptr_lamb_y = pb.dptr_lamb_y;
	pList.dptr_lamb_z = pb.dptr_lamb_z;

	cudaError_t err;

	//int nthreadx = 16;
	//int nthready = 16;
	//int nDimx = pb.py / nthreadx;
	//int nDimy = pb.pz / nthready;
	//if (pb.py % nthreadx != 0) nDimx++;
	//if (pb.pz % nthready != 0) nDimy++;
	//dim3 nThread(nthreadx, nthready);
	//dim3 nDim(nDimx, nDimy);
	dim3 nDim(pb.my, pb.pz);
	//computeLambVectorKernel<<<pb.nDim,pb.nThread>>>(pList, pb.px, pb.py, pb.pz, pb.Ro);
	computeLambVectorKernel<<<nDim,pb.mx>>>(pList, pb.px, pb.py, pb.pz, pb.Ro);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);

	//safeCudaFree(pb.dptr_u.ptr);
	//safeCudaFree(pb.dptr_v.ptr);
	//safeCudaFree(pb.dptr_w.ptr);
	//safeCudaFree(pb.dptr_omega_x.ptr);
	//safeCudaFree(pb.dptr_omega_y.ptr);
	//safeCudaFree(pb.dptr_omega_z.ptr);
	pb.dptr_u.ptr = NULL;
	pb.dptr_v.ptr = NULL;
	pb.dptr_w.ptr = NULL;
	myCudaFree(pb.dptr_omega_z, XYZ_3D);
	myCudaFree(pb.dptr_omega_y, XYZ_3D);
	myCudaFree(pb.dptr_omega_x, XYZ_3D);

	return 0;
}

__host__ int rhsNonlinear(problem & pb)
{
	// get datas
	cudaPitchedPtrList tPlist(pb,TRANSPOSED);
/*
	int dim[3];
	int tDim[3];
	dim[0] = pb.mx;
	dim[1] = pb.my;
	dim[2] = pb.mz;
	tDim[0] = pb.mz;
	tDim[1] = pb.mx;
	tDim[2] = pb.my;
*/
	cudaError_t err;
	int nthreadx = 16;
	int nthready = 16;
	const int hnx = pb.mx/ 3 * 2 / 2+1;
	const int ny = pb.my / 3 * 2;
	int nDimx = hnx / nthreadx;
	int nDimy = ny / nthready;
	if (hnx % nthreadx != 0) nDimx++;
	if (ny%nthready != 0)nDimy++;
	dim3 nThread(nthreadx, nthready);
	dim3 nDim(nDimx, nDimy);

	//rhsNonlinearKernel<<<nDim,nThread>>>(tPlist, pb.mx, pb.my, pb.mz, pb.aphi, pb.beta);
	rhsNonlinearKernel<<<dim3(hnx-1,ny),pb.nz>>>(tPlist, pb.mx, pb.my, pb.mz, pb.aphi, pb.beta);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);

	/*
	//data type mismatch here!
	res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)pb.dptr_tLamb_x.ptr,
		(CUFFTCOMPLEX*)pb.dptr_tLamb_x.ptr,CUFFT_INVERSE); 
	ASSERT(res == CUFFT_SUCCESS);
	res = CUFFTEXEC_C2C(planZ_pad, (CUFFTCOMPLEX*)pb.dptr_tLamb_z.ptr,
		(CUFFTCOMPLEX*)pb.dptr_tLamb_z.ptr,CUFFT_INVERSE);
	ASSERT(res == CUFFT_SUCCESS);

	int nthreadx = 7;
	int nthready = 6;
	ASSERT((pb.mx/2+1) % nthreadx == 0);
	ASSERT(pb.my % nthready == 0);
	dim3 thread_num(nthreadx, nthready);
	normalize << <1, thread_num >> >
		(pb.dptr_tLamb_x, pb.mz * 2, pb.mx / 2 + 1, pb.my, 1.0 / pb.mz);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess); 
	
	normalize << <1, thread_num >> >
		(pb.dptr_tLamb_z, pb.mz * 2, pb.mx / 2 + 1, pb.my, 1.0 / pb.mz);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);

	//swap of old RHS value and new
	swap(pb.nonlinear_v, pb.nonlinear_v_p);
	swap(pb.nonlinear_omega_y, pb.nonlinear_omega_y_p);

	//copy RHS from device to host.
	size_t tSize = pb.dptr_tLamb_x.pitch*(pb.mx/2+1)*pb.my;
	err = cudaMemcpy(pb.nonlinear_v, pb.dptr_tLamb_x.ptr, tSize, cudaMemcpyDeviceToHost);
	ASSERT(err == cudaSuccess);
	err = cudaMemcpy(pb.nonlinear_omega_y, pb.dptr_tLamb_z.ptr, tSize, cudaMemcpyDeviceToHost);
	ASSERT(err == cudaSuccess);
	*/

	return int();
}
__global__ void rhsNonlinearKernel(cudaPitchedPtrList plist,
	int mx, int my, int mz, real alpha, real beta)
{
	int kx = blockIdx.x;
	int ky = blockIdx.y; 
	int kz = threadIdx.x;
	int pitch = plist.dptr_lamb_x.pitch;
	//int pz = mz / 2 + 1;
	int nz = mz / 4 + 1;

	// skip the k=0 mode
	//if (kx == 0 && ky == 0) return;

	if (kx == 0 && ky == 0) return;

	//skip non-necessary wave numbers.
	const int nx = mx / 3 * 2;
	const int ny = my / 3 * 2;
	if (kx >= nx / 2 + 1)return;
	if (ky >= ny) return;

	real ialpha = real(kx) / alpha;
	real ibeta = real(ky) / beta;	
	if (ky >= ny / 2 + 1) {
		ibeta = real(ky - ny) / beta;
	}

	real kmn = ialpha*ialpha + ibeta*ibeta;

	//temp vector for derivative real part
	__shared__ real tdz_re[MAX_NZ];
	//temp result u real part
	real tres_u_re;
	//temp result w real part
	real tres_w_re;

	//temp vector for derivative imaginary part
	__shared__ real tdz_im[MAX_NZ];
	//temp result u imaginary part
	real tres_u_im;
	//temp result v imaginary part part
	real tres_w_im;

	

	// the following variables are in spectral space
	//complex* dp_u = (complex*)plist.dptr_u.ptr; //actually dptr_tu, so on...
	//complex* dp_v = (complex*)plist.dptr_v.ptr;
	//complex* dp_w = (complex*)plist.dptr_w.ptr;

	complex* dp_lamb_x = (complex*)plist.dptr_lamb_x.ptr;
	complex* dp_lamb_y = (complex*)plist.dptr_lamb_y.ptr;
	complex* dp_lamb_z = (complex*)plist.dptr_lamb_z.ptr;

	// change location of pointers.
	size_t dist = (kx + (nx/2+1)*ky)*pitch / sizeof(complex);
	//dp_u = dp_u + dist;
	//dp_v = dp_v + dist;
	//dp_w = dp_w + dist;
	dp_lamb_x = dp_lamb_x + dist;
	dp_lamb_y = dp_lamb_y + dist;
	dp_lamb_z = dp_lamb_z + dist;

	int i = kz;

	complex cache_lamb_x = dp_lamb_x[i];
	complex cache_lamb_y = dp_lamb_y[i];
	complex cache_lamb_z = dp_lamb_z[i];

		tdz_re[i] = cache_lamb_x.re;
		tdz_im[i] = cache_lamb_x.im;

	ddz_sm(tdz_re, nz, i);
	ddz_sm(tdz_im, nz, i);

		tres_u_re = kmn*cache_lamb_z.re - ialpha * tdz_im[i];
		tres_u_im = kmn*cache_lamb_z.im + ialpha * tdz_re[i];

		tdz_re[i] = cache_lamb_y.re;
		tdz_im[i] = cache_lamb_y.im;

	ddz_sm(tdz_re, nz, i);
	ddz_sm(tdz_im, nz, i);

		tres_u_re = tres_u_re - ibeta*tdz_im[i];
		tres_u_im = tres_u_im + ibeta*tdz_re[i];

		tres_w_re = -1.0*ialpha*cache_lamb_y.im + ibeta*cache_lamb_x.im;
		tres_w_im =      ialpha*cache_lamb_y.re - ibeta*cache_lamb_x.re;
	// end of computation of rhs term.


	// the results are stored in lamb_x and lamb_z in spectral space
		dp_lamb_x[i].re = tres_u_re;
		dp_lamb_x[i].im = tres_u_im;
		dp_lamb_y[i].re = tres_w_re;
		dp_lamb_y[i].im = tres_w_im;
}

__host__ int addCoriolisForce(problem & pb)
{
	// this operation will be done in computeLamb() 
	return 0;
}


__global__ void addMeanFlowKernel(cudaPitchedPtr ptr, int px, int py, int pz) {
	int iy = threadIdx.x + blockDim.x*blockIdx.x;
	int iz = threadIdx.y + blockDim.y*blockIdx.y;
	if (iy >= py || iz >= pz) return;
	size_t id = py*iz + iy;

	const real PI = 4.0*atan(1.0);
	real z = cos((real)iz / (pz - 1)*PI);
	real mean_u = 0.5*(z + 1);

	real* dp_u;
	int pitch = ptr.pitch;
	dp_u = (real*)ptr.ptr + pitch/sizeof(real) * id;
	for (int i = 0; i < px; i++) {
		dp_u[i] = dp_u[i] + mean_u;
	}
}
//
//__global__ void computeLambVectorKernekl _non(cudaPitchedPtrList ptrList,
//	const int mx, const int my, const int pz) {
//	//可以合并几个lamb矢量计算到一个kernel中
//	int ky = threadIdx.x + blockDim.x*blockIdx.x;
//	int kz = threadIdx.y + blockDim.y*blockIdx.y;
//	//const int py = my * 2 / 3;
//	//const int pz = mz / 2 + 1;
//	if (ky >= my || kz >= pz) return;
//
//
//	int id = my*kz + ky;
//	cudaPitchedPtr& ptrU = ptrList.dptr_u;
//	cudaPitchedPtr& ptrV = ptrList.dptr_v;
//	cudaPitchedPtr& ptrW = ptrList.dptr_w;
//	cudaPitchedPtr& ptrOmegaX = ptrList.dptr_omega_x;
//	cudaPitchedPtr& ptrOmegaY = ptrList.dptr_omega_y;
//	cudaPitchedPtr& ptrOmegaZ = ptrList.dptr_omega_z;
//	cudaPitchedPtr& ptrLambX = ptrList.dptr_lamb_x;
//	cudaPitchedPtr& ptrLambY = ptrList.dptr_lamb_y;
//	cudaPitchedPtr& ptrLambZ = ptrList.dptr_lamb_z;
//
//	//compute the location of data in each thread.
//	int pitch = ptrV.pitch;
//	real* pU = (real*)((char*)ptrU.ptr + pitch * id);
//	real* pV = (real*)((char*)ptrV.ptr + pitch * id);
//	real* pW = (real*)((char*)ptrW.ptr + pitch * id);
//	real* pOmegaX = (real*)((char*)ptrOmegaX.ptr + pitch * id);
//	real* pOmegaY = (real*)((char*)ptrOmegaY.ptr + pitch * id);
//	real* pOmegaZ = (real*)((char*)ptrOmegaZ.ptr + pitch * id);
//	real* pLambX = (real*)((char*)ptrLambX.ptr + pitch * id);
//	real* pLambY = (real*)((char*)ptrLambY.ptr + pitch * id);
//	real* pLambZ = (real*)((char*)ptrLambZ.ptr + pitch * id);
//
//	//execute the computation
//	computeLambDevice(pU, pV, pW, pOmegaX, pOmegaY, pOmegaZ,
//		pLambX, pLambY, pLambZ, mx);
//}

__global__ void computeLambVectorKernel(cudaPitchedPtrList ptrList,
	const int mx, const int my, const int pz, real Ro) {
	//可以合并几个lamb矢量计算到一个kernel中
	int ky = blockIdx.x;
	int kz = blockIdx.y;
	int kx = threadIdx.x;
	//const int py = my * 2 / 3;
	//const int pz = mz / 2 + 1;
	if (ky >= my || kz >= pz || kx>=mx) return;


	int id = my*kz + ky;
	cudaPitchedPtr& ptrU = ptrList.dptr_u;
	cudaPitchedPtr& ptrV = ptrList.dptr_v;
	cudaPitchedPtr& ptrW = ptrList.dptr_w;
	cudaPitchedPtr& ptrOmegaX = ptrList.dptr_omega_x;
	cudaPitchedPtr& ptrOmegaY = ptrList.dptr_omega_y;
	cudaPitchedPtr& ptrOmegaZ = ptrList.dptr_omega_z;
	cudaPitchedPtr& ptrLambX = ptrList.dptr_lamb_x;
	cudaPitchedPtr& ptrLambY = ptrList.dptr_lamb_y;
	cudaPitchedPtr& ptrLambZ = ptrList.dptr_lamb_z;

	//compute the location of data in each thread.
	int pitch = ptrV.pitch;
	real* pU = (real*)((char*)ptrU.ptr + pitch * id) + kx;
	real* pV = (real*)((char*)ptrV.ptr + pitch * id) + kx;
	real* pW = (real*)((char*)ptrW.ptr + pitch * id) + kx;
	real* pOmegaX = (real*)((char*)ptrOmegaX.ptr + pitch * id) + kx;
	real* pOmegaY = (real*)((char*)ptrOmegaY.ptr + pitch * id) + kx;
	real* pOmegaZ = (real*)((char*)ptrOmegaZ.ptr + pitch * id) + kx;
	real* pLambX = (real*)((char*)ptrLambX.ptr + pitch * id) + kx;
	real* pLambY = (real*)((char*)ptrLambY.ptr + pitch * id) + kx;
	real* pLambZ = (real*)((char*)ptrLambZ.ptr + pitch * id) + kx;

	//execute the computation
	computeLambDevice(pU, pV, pW, pOmegaX, pOmegaY, pOmegaZ,
		pLambX, pLambY, pLambZ, mx, Ro);
}

__device__ void computeLambDevice(real* pU, real* pV, real* pW,
	real* pOmegaX, real* pOmegaY, real* pOmegaZ,
	real* pLambX, real* pLambY, real* pLambZ, int mx, real Ro) {
	int  i = 0;
	pLambX[i] = pOmegaY[i] * pW[i] - pOmegaZ[i] * pV[i] - Ro*pW[i];
	pLambY[i] = pOmegaZ[i] * pU[i] - pOmegaX[i] * pW[i];
	pLambZ[i] = pOmegaX[i] * pV[i] - pOmegaY[i] * pU[i] + Ro*pU[i];
}
//
//__device__ void computeLambDevice_non(real* pU, real* pV, real* pW,
//	real* pOmegaX, real* pOmegaY, real* pOmegaZ,
//	real* pLambX, real* pLambY, real* pLambZ, int mx) {
//
//	real tU[CACHE_SIZE];
//	real tV[CACHE_SIZE];
//	real tW[CACHE_SIZE];
//	
//	real tOmegaX[CACHE_SIZE];
//	real tOmegaY[CACHE_SIZE];
//	real tOmegaZ[CACHE_SIZE];
//
//	real tLambX[CACHE_SIZE];
//	real tLambY[CACHE_SIZE];
//	real tLambZ[CACHE_SIZE];
//
//	int nCycle = mx / CACHE_SIZE + 1;
//	ASSERT(mx < CACHE_SIZE);
//
//	//cycles except the last cycle.
//	for (int iCycle = 0; iCycle < nCycle - 1; iCycle++) {
//		for (int i = 0; i < CACHE_SIZE; i++) {
//			int index = iCycle*CACHE_SIZE + i;
//			tU[i] = pU[index];
//			tV[i] = pV[index];
//			tW[i] = pW[index];
//			tOmegaX[i] = pOmegaX[index];
//			tOmegaY[i] = pOmegaY[index];
//			tOmegaZ[i] = pOmegaZ[index];
//		}
//		for (int i = 0; i < CACHE_SIZE; i++) {
//			tLambX[i] = tOmegaY[i] * tW[i] - tOmegaZ[i] * tV[i];
//			tLambY[i] = tOmegaZ[i] * tU[i] - tOmegaX[i] * tW[i];
//			tLambZ[i] = tOmegaX[i] * tV[i] - tOmegaY[i] * tU[i];
//		}
//		//seperate reading and writing action to avoid memory conflicts
//		for (int i = 0; i < CACHE_SIZE; i++) {
//			int index = iCycle*CACHE_SIZE + i;
//			pLambX[index] = tLambX[i];
//			pLambY[index] = tLambY[i];
//			pLambZ[index] = tLambZ[i];
//		}
//	}
//
//	ASSERT(nCycle >= 1);
//	int nLast = mx%CACHE_SIZE;
//	//for the last part.
//	for (int i = 0; i < nLast; i++) {
//		int index = (nCycle-1)*CACHE_SIZE + i;
//		tU[i] = pU[index];
//		tV[i] = pV[index];
//		tW[i] = pW[index];
//		tOmegaX[i] = pOmegaX[index];
//		tOmegaY[i] = pOmegaY[index];
//		tOmegaZ[i] = pOmegaZ[index];
//	}
//	for (int i = 0; i < nLast; i++) {
//		tLambX[i] = tOmegaY[i] * tW[i] - tOmegaZ[i] * tV[i];
//		tLambY[i] = tOmegaZ[i] * tU[i] - tOmegaX[i] * tW[i];
//		tLambZ[i] = tOmegaX[i] * tV[i] - tOmegaY[i] * tU[i];
//	}
//	//seperate reading and writing action to avoid memory conflicts
//	for (int i = 0; i < nLast; i++) {
//		int index = (nCycle - 1)*CACHE_SIZE + i;
//		pLambX[index] = tLambX[i];
//		pLambY[index] = tLambY[i];
//		pLambZ[index] = tLambZ[i];
//	}
//
//}

__host__ void saveZeroWaveLamb(problem & pb)
{
	swap(pb.lambx0, pb.lambx0_p);
	swap(pb.lambz0, pb.lambz0_p);

	complex* lambx = (complex*)pb.dptr_tLamb_x.ptr;
	complex* lambz = (complex*)pb.dptr_tLamb_y.ptr;	//change of y,z definition here!
	cudaError err;
	err = cudaMemcpy(pb.lambx0, lambx, pb.nz * sizeof(complex), cudaMemcpyDeviceToHost);
	ASSERT(err == cudaSuccess);
	err = cudaMemcpy(pb.lambz0, lambz, pb.nz * sizeof(complex), cudaMemcpyDeviceToHost);
	ASSERT(err == cudaSuccess);

}