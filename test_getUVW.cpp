#include "test_getUVW.h"
#include "velocity.h"
#include "operation.h"
#include "solver.h"
#include "cuRPCF.h"
#include "transform.cuh"
#include "rhs.cuh"

#include <iostream>
using namespace std;
TestResult test_getUVW() {
	RPCF_Paras para("parameter.txt");
	problem pb(para);
	initCUDA(pb);
	initFFT(pb);
	initSolver(pb);
	initFlow(pb);
	
	get_rhs_v(pb);
	RPCF::write_3d_to_file("rhs_v.txt", (real*)pb.rhs_v,
		pb.tPitch, 2*pb.mz, (pb.mx / 2 + 1), pb.my);
	
	return TestSuccess;
}