#include <iostream>
#include "../include/data.h"
#include "../include/operation.h"
#include "../include/RPCFKernels.cuh"
#include "../include/transform.cuh"
#include "../include/nonlinear.cuh"
#include "../include/solver.h"
#include "../include/parameters.h"

using namespace std;

__host__ int transpose(DIRECTION dir, cudaPitchedPtr input,
	cudaPitchedPtr output, int* indim, int* outdim);

int _main(int args, char** argv) {
	RPCF_Paras config("parameter.txt");
	problem pb(config);
	allocDeviceMem(pb);
	initFFT(pb);
	allocHostMemory(pb);
	initFlow(pb);
	nextStep(pb);
	return 0;
}

//int _main2(int args, char** argv) {
//	problem pb;
//	cout << pb.nx << endl;
//	initCUDA(pb);
//	initFlow(pb);
//	computeNonlinear(pb);
//	cudaDeviceSynchronize();
//	pb.memcpy_device_to_host();
//	RPCF::write_3d_to_file("init.txt", pb.hptr_u, pb.dptr_u.pitch,
//		pb.nx, pb.ny, pb.nz);
//	cout << "end" << endl;
//	return 0;
//}