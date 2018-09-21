#include <iostream>
#include "data.h"
#include "operation.h"
#include "RPCFKernels.cuh"
#include "transform.cuh"
#include "nonlinear.cuh"
#include "solver.h"
#include "parameters.h"

using namespace std;

__host__ int transpose(DIRECTION dir, cudaPitchedPtr input,
	cudaPitchedPtr output, int* indim, int* outdim);

int main(int args, char** argv) {
	RPCF_Paras config("parameter.txt");
	problem pb(config);
	initCUDA(pb);
	initFFT(pb);
	initSolver(pb);
	initFlow(pb);
	nextStep(pb);
	return 0;
}

int _main2(int args, char** argv) {
	problem pb;
	cout << pb.nx << endl;
	initCUDA(pb);
	initFlow(pb);
	computeNonlinear(pb);
	cudaDeviceSynchronize();
	pb.memcpy_device_to_host();
	RPCF::write_3d_to_file("init.txt", pb.hptr_u, pb.dptr_u.pitch,
		pb.nx, pb.ny, pb.nz);
	cout << "end" << endl;
	return 0;
}