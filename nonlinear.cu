#include "nonlinear.cuh"
#include "ASSERT.h"
#include "cuRPCF.h"

__host__ int addMeanFlow(problem& pb);

__host__ int addCoriolisForce(problem& pb);

__device__ void computeLambDevice(real* pU, real* pV, real* pW,
	real* pOmegaX, real* pOmegaY, real* pOmegaZ,
	real* pLambX, real* pLambY, real* pLambZ, int mz);

__global__ void addMeanFlowKernel(cudaPitchedPtr ptr, int px, int py, int pz);
__global__ void computeLambVectorKernel(cudaPitchedPtrList pList, 
	int mx, int my, int mz);

__host__ void saveZeroWaveLamb(problem& pb);


// Get the nonliear part of RHS
__host__ int getNonlinear(problem& pb) {

	//TODO: move this operation to the outside of the function.
	// spec --> phys, tPtr(z,x,y) --> Ptr(x,y,z)
	//transform(BACKWARD, pb);

	// use Ptr in pb
	addMeanFlow(pb);

	computeLambVector(pb);

	// phys --> spec, Ptr(x,y,z) --> tPtr(z,x,y)
	// the forward transform only deals with lamb vectors !
	// see the implementation of the function for further details.
	transform(FORWARD, pb);

	addCoriolisForce(pb);

	// this operation will be finished in get_rhs_v;
	//saveZeroWaveLamb(pb);

	// Get the right hand side (RHS) part of the equation.
	// transform the nonlinear RHS part into phys space
	// transform(BACKWARD, pb);
	// the transfomr is done inside the function.
	rhsNonlinear(pb);

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
	addMeanFlowKernel <<<pb.nDim, pb.nThread>>>(pb.dptr_u, pb.mx, pb.my, pb.mz);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);
	return int();
}

__host__ int computeLambVector(problem & pb)
{
	const int mx = pb.mx;
	const int my = pb.my;
	const int mz = pb.mz;

	cudaExtent& extent = pb.extent;
	//make_cudaExtent(
	//	2 * (pb.mx / 2 + 1) * sizeof(real), pb.my, pb.mz);
	cuCheck(cudaMalloc3D(&(pb.dptr_lamb_x), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_lamb_y), extent), "allocate");
	cuCheck(cudaMalloc3D(&(pb.dptr_lamb_z), extent), "allocate");

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

	computeLambVectorKernel<<<pb.nDim,pb.nThread>>>(pList, pb.px, pb.py, pb.pz);
	err = cudaDeviceSynchronize();
	ASSERT(err == cudaSuccess);

	safeCudaFree(pb.dptr_u.ptr);
	safeCudaFree(pb.dptr_v.ptr);
	safeCudaFree(pb.dptr_w.ptr);
	safeCudaFree(pb.dptr_omega_x.ptr);
	safeCudaFree(pb.dptr_omega_y.ptr);
	safeCudaFree(pb.dptr_omega_z.ptr);

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
	int nDimx = (pb.mx / 2 + 1) / nthreadx;
	int nDimy = pb.my / nthready;
	if ((pb.mx / 2 + 1) % nthreadx != 0) nDimx++;
	if (pb.my%nthready != 0)nDimy++;
	dim3 nThread(nthreadx, nthready);
	dim3 nDim(nDimx, nDimy);

	rhsNonlinearKernel<<<nDim,nThread>>>(tPlist, pb.mx, pb.my, pb.mz, pb.aphi, pb.beta);
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
	
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	int ky = threadIdx.y + blockDim.y*blockIdx.y; 
	int pitch = plist.dptr_lamb_x.pitch;
	int pz = mz / 2 + 1;

	// skip the k=0 mode
	//if (kx == 0 && ky == 0) return;
	if (kx >= (mx / 2 + 1) || ky >= my) return;
	if (kx == 0 && ky == 0) return;

	real ialpha = real(kx) / alpha;
	real ibeta = real(ky) / beta;	
	if (ky >= my / 2 + 1) {
		ibeta = real(ky - my) / beta;
	}

	real kmn = ialpha*ialpha + ibeta*ibeta;

	//temp vector for derivative real part
	real tdz_re[MAX_NZ * 4];
	//temp result u real part
	real tres_u_re[MAX_NZ * 4];
	//temp result w real part
	real tres_w_re[MAX_NZ * 4];

	//temp vector for derivative imaginary part
	real tdz_im[MAX_NZ * 4];
	//temp result u imaginary part
	real tres_u_im[MAX_NZ * 4];
	//temp result v imaginary part part
	real tres_w_im[MAX_NZ * 4];


	// the following variables are in spectral space
	//complex* dp_u = (complex*)plist.dptr_u.ptr; //actually dptr_tu, so on...
	//complex* dp_v = (complex*)plist.dptr_v.ptr;
	//complex* dp_w = (complex*)plist.dptr_w.ptr;

	complex* dp_lamb_x = (complex*)plist.dptr_lamb_x.ptr;
	complex* dp_lamb_y = (complex*)plist.dptr_lamb_y.ptr;
	complex* dp_lamb_z = (complex*)plist.dptr_lamb_z.ptr;

	// change location of pointers.
	size_t dist = (kx + (mx/2+1)*ky)*pitch / sizeof(complex);
	//dp_u = dp_u + dist;
	//dp_v = dp_v + dist;
	//dp_w = dp_w + dist;
	dp_lamb_x = dp_lamb_x + dist;
	dp_lamb_y = dp_lamb_y + dist;
	dp_lamb_z = dp_lamb_z + dist;

	for (int i = 0; i < pz; i++) {
		tdz_re[i] = dp_lamb_x[i].re;
		tdz_im[i] = dp_lamb_x[i].im;
	}

	ddz(tdz_re, pz);
	ddz(tdz_im, pz);

	for (int i = 0; i < pz; i++) {
		tres_u_re[i] = kmn*dp_lamb_z[i].re - ialpha * tdz_im[i];
		tres_u_im[i] = kmn*dp_lamb_z[i].im + ialpha * tdz_re[i];
	}

	for (int i = 0; i < pz; i++) {
		tdz_re[i] = dp_lamb_y[i].re;
		tdz_im[i] = dp_lamb_y[i].im;
	}

	ddz(tdz_re, pz);
	ddz(tdz_im, pz);

	for (int i = 0; i < pz; i++) {
		tres_u_re[i] = tres_u_re[i] - ibeta*tdz_im[i];
		tres_u_im[i] = tres_u_im[i] + ibeta*tdz_re[i];

		tres_w_re[i] = -1*ialpha*dp_lamb_y[i].im + ibeta*dp_lamb_x[i].im;
		tres_w_im[i] =   ialpha*dp_lamb_y[i].re - ibeta*dp_lamb_x[i].re;
	}
	// end of computation of rhs term.


	// the results are stored in lamb_x and lamb_z in spectral space
	for (int i = 0; i < pz; i++) {
		dp_lamb_x[i].re = tres_u_re[i];
		dp_lamb_x[i].im = tres_u_im[i];
		dp_lamb_y[i].re = tres_w_re[i];
		dp_lamb_y[i].im = tres_w_im[i];
	}
}

__host__ int addCoriolisForce(problem & pb)
{
	
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

__global__ void computeLambVectorKernel(cudaPitchedPtrList ptrList,
	const int mx, const int my,const int mz) {
	//可以合并几个lamb矢量计算到一个kernel中
	int ky = threadIdx.x + blockDim.x*blockIdx.x;
	int kz = threadIdx.y + blockDim.y*blockIdx.y;
	if (ky >= my || kz >= mz) return;
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
	real* pU = (real*)((char*)ptrU.ptr + pitch * id);
	real* pV = (real*)((char*)ptrV.ptr + pitch * id);
	real* pW = (real*)((char*)ptrW.ptr + pitch * id);
	real* pOmegaX = (real*)((char*)ptrOmegaX.ptr + pitch * id);
	real* pOmegaY = (real*)((char*)ptrOmegaY.ptr + pitch * id);
	real* pOmegaZ = (real*)((char*)ptrOmegaZ.ptr + pitch * id);
	real* pLambX = (real*)((char*)ptrLambX.ptr + pitch * id);
	real* pLambY = (real*)((char*)ptrLambY.ptr + pitch * id);
	real* pLambZ = (real*)((char*)ptrLambZ.ptr + pitch * id);

	//execute the computation
	computeLambDevice(pU, pV,  pW, pOmegaX, pOmegaY,  pOmegaZ,
		pLambX, pLambY, pLambZ, mx);
}

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

__device__ void computeLambDevice(real* pU, real* pV, real* pW,
	real* pOmegaX, real* pOmegaY, real* pOmegaZ,
	real* pLambX, real* pLambY, real* pLambZ, int mx) {

	real tU[CACHE_SIZE];
	real tV[CACHE_SIZE];
	real tW[CACHE_SIZE];
	
	real tOmegaX[CACHE_SIZE];
	real tOmegaY[CACHE_SIZE];
	real tOmegaZ[CACHE_SIZE];

	real tLambX[CACHE_SIZE];
	real tLambY[CACHE_SIZE];
	real tLambZ[CACHE_SIZE];

	int nCycle = mx / CACHE_SIZE + 1;
	ASSERT(mx < CACHE_SIZE);

	//cycles except the last cycle.
	for (int iCycle = 0; iCycle < nCycle - 1; iCycle++) {
		for (int i = 0; i < CACHE_SIZE; i++) {
			int index = iCycle*CACHE_SIZE + i;
			tU[i] = pU[index];
			tV[i] = pV[index];
			tW[i] = pW[index];
			tOmegaX[i] = pOmegaX[index];
			tOmegaY[i] = pOmegaY[index];
			tOmegaZ[i] = pOmegaZ[index];
		}
		for (int i = 0; i < CACHE_SIZE; i++) {
			tLambX[i] = tOmegaY[i] * tW[i] - tOmegaZ[i] * tV[i];
			tLambY[i] = tOmegaZ[i] * tU[i] - tOmegaX[i] * tW[i];
			tLambZ[i] = tOmegaX[i] * tV[i] - tOmegaY[i] * tU[i];
		}
		//seperate reading and writing action to avoid memory conflicts
		for (int i = 0; i < CACHE_SIZE; i++) {
			int index = iCycle*CACHE_SIZE + i;
			pLambX[index] = tLambX[i];
			pLambY[index] = tLambY[i];
			pLambZ[index] = tLambZ[i];
		}
	}

	ASSERT(nCycle >= 1);
	int nLast = mx%CACHE_SIZE;
	//for the last part.
	for (int i = 0; i < nLast; i++) {
		int index = (nCycle-1)*CACHE_SIZE + i;
		tU[i] = pU[index];
		tV[i] = pV[index];
		tW[i] = pW[index];
		tOmegaX[i] = pOmegaX[index];
		tOmegaY[i] = pOmegaY[index];
		tOmegaZ[i] = pOmegaZ[index];
	}
	for (int i = 0; i < nLast; i++) {
		tLambX[i] = tOmegaY[i] * tW[i] - tOmegaZ[i] * tV[i];
		tLambY[i] = tOmegaZ[i] * tU[i] - tOmegaX[i] * tW[i];
		tLambZ[i] = tOmegaX[i] * tV[i] - tOmegaY[i] * tU[i];
	}
	//seperate reading and writing action to avoid memory conflicts
	for (int i = 0; i < nLast; i++) {
		int index = (nCycle - 1)*CACHE_SIZE + i;
		pLambX[index] = tLambX[i];
		pLambY[index] = tLambY[i];
		pLambZ[index] = tLambZ[i];
	}

}