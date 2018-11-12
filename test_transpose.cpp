#include "test_transpose.h"
#include "data.h"
#include "cuRPCF.h"
#include "transform.cuh"
#include "transpose.cuh"
#include <cassert>

TestResult test_transpose()
{
	RPCF_Paras para("parameter.txt");
	problem pb(para);
	initCUDA(pb);
	initFFT(pb);

	int mx = pb.mx;
	int my = pb.my;
	int mz = pb.mz;
	int pitch = pb.pitch;
	int tPitch = pb.tPitch;

	size_t size = pb.pSize;
	size_t tSize = pb.tSize;


	cudaExtent extent = make_cudaExtent(sizeof(complex)*(mx/2+1), my, pb.pz);
	cudaExtent tExtent = make_cudaExtent(sizeof(complex)*mz, pb.nx, pb.ny);

	cudaPitchedPtr Ptr, tPtr;
	cuCheck(cudaMalloc3D(&Ptr, extent), "mem alloc");
	cuCheck(cudaMalloc3D(&tPtr, tExtent), "mem alloc");

	pitch = Ptr.pitch;
	tPitch = tPtr.pitch;

	complex* data1 = (complex*)malloc(size);
	complex* data2 = (complex*)malloc(tSize);
	complex* tData = (complex*)malloc(tSize);
	

	for (int i = 0; i < pb.nx/2+1; i++) {
		for (int j = 0; j < pb.ny; j++) {
			for (int k = 0; k < pb.pz; k++) {
				int ky = j;
				if (ky > pb.ny / 2) ky = j + (pb.my - pb.ny);
				size_t inc1 = pitch / sizeof(complex)*(my*k + ky) + i;
				size_t inc2 = tPitch / sizeof(complex)*((pb.nx/2+1)*j+i) + k;
				assert(inc1 <= size / sizeof(complex));
				assert(inc2 <= tSize / sizeof(complex));
				data1[inc1] = 100 * i + j + 0.01*k;
				data2[inc2] = 100 * i + j + 0.01*k;
			}
		}
	}

	cuCheck(cudaMemcpy(Ptr.ptr, data1, size, cudaMemcpyHostToDevice), "mem cpy");

	int dim[3] = { mx,my,mz };
	int tDim[3] = { mz,mx,my };
	cuda_transpose(FORWARD, Ptr, tPtr,dim, tDim);

	cuCheck(cudaMemcpy(tData, tPtr.ptr, tSize, cudaMemcpyDeviceToHost), "mem cpy");

	for (int i = 0; i < pb.nx / 2 + 1; i++) {
		for (int j = 0; j < pb.ny; j++) {
			for (int k = 0; k < pb.pz; k++) {
				int ky = j;
				if (ky > pb.ny / 2) ky = j + (pb.my - pb.ny);
				size_t inc1 = pitch / sizeof(complex)*(my*k + ky) + i;
				size_t inc2 = tPitch / sizeof(complex)*((pb.nx / 2 + 1)*j + i) + k;
				assert(isEqual(data1[inc1].re, tData[inc2].re));
			}
		}
	}

	cuda_transpose(BACKWARD, Ptr, tPtr, dim, tDim);
	cuCheck(cudaMemcpy(data1, Ptr.ptr, size, cudaMemcpyDeviceToHost), "mem cpy");
	for (int i = 0; i < pb.nx / 2 + 1; i++) {
		for (int j = 0; j < pb.ny; j++) {
			for (int k = 0; k < pb.pz; k++) {
				int ky = j;
				if (ky > pb.ny / 2) ky = j + (pb.my - pb.ny);
				size_t inc1 = pitch / sizeof(complex)*(my*k + ky) + i;
				size_t inc2 = tPitch / sizeof(complex)*((pb.nx / 2 + 1)*j + i) + k;
				assert(isEqual(data1[inc1].re, tData[inc2].re));
			}
		}
	}

	return TestSuccess;
}
