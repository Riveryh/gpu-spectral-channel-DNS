#include "test_dt_0.h"
#include "cuRPCF.h"
#include "solver.h"
#include "output.h"
#include <cassert>
#include "statistics.h"
#include <time.h>  
#include <stdio.h>  

TestResult test_dt_0() {
	RPCF_Paras paras("parameter.txt");
	//paras.numPara.dt = 0.0;
	problem pb(paras);
	initCUDA(pb);
	initFFT(pb);
	initSolver(pb);
	initFlow(pb);

	output_velocity(pb);

	complex* tv = (complex*)malloc(pb.tSize);
	
	nextStep(pb);
	//nextStep(pb);
	initCUDA(pb);
	initFlow(pb);
	nextStep(pb);
	output_velocity(pb);

	//complex* tv2 = pb.rhs_v;
	complex* tv2 = (complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(tv, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	
	clock_t start_time, end_time;
	double cost;
	double total_cost = 0.0;
	int count = 0;
	for (int i = 0; i < 1000; i++) {
		std::cout << "step: " << i << std::endl;
		start_time = clock();
		nextStep(pb);
		//statistics(pb);
		end_time = clock();
		cost = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		total_cost += cost;
		count++;
		if (i % 10 == 0) output_velocity(pb);
		std::cout << "time_cost:" << cost << std::endl;
		std::cout << "mean time cost:" << total_cost / count << std::endl;
	}
	output_velocity(pb);
	cuCheck(cudaMemcpy(tv2, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");

	return TestSuccess;

	for (int ix = 0; ix < pb.mx / 2 + 1; ix++) {
		for (int iy = 0; iy < pb.my; iy++) {
			for (int iz = 0; iz < pb.nz; iz++) {
				size_t inc = pb.tPitch / sizeof(complex)
					*((pb.mx / 2 + 1)*iy + ix);
				assert(isEqual(tv[inc].re, tv2[inc].re));
				assert(isEqual(tv[inc].im, tv2[inc].im));
			}
		}
	}

	return TestSuccess;
}
