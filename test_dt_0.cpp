#include "test_dt_0.h"
#include "cuRPCF.h"
#include "solver.h"
#include "output.h"
#include <cassert>
#include "statistics.h"
#include <time.h>  
#include <stdio.h>  
#include <iostream>
using namespace std;

TestResult test_dt_0() {
	RPCF_Paras paras("parameter.txt");
	//paras.numPara.dt = 0.0;
	problem pb(paras);
	cout << "init cuda" << endl;
	initCUDA(pb);
	cout << "init fft" << endl;
	initFFT(pb);
	cout << "init solver" << endl;
	initSolver(pb);
	cout << "init flow" << endl;
	initFlow(pb);

	cout << "output flow" << endl;

	//output_velocity(pb);
	//output_velocity(pb);

	complex* tv = (complex*)malloc(pb.tSize);


	cout << "first step" << endl;
	nextStep(pb);
	//safeCudaFree(pb.dptr_tu.ptr);
	//safeCudaFree(pb.dptr_tv.ptr);
	//safeCudaFree(pb.dptr_tw.ptr);
	//safeCudaFree(pb.dptr_tomega_x.ptr);
	//safeCudaFree(pb.dptr_tomega_y.ptr);
	//safeCudaFree(pb.dptr_tomega_z.ptr);
	destroyMyCudaMalloc();

	//nextStep(pb);

	cout << "init cuda" << endl;
	initCUDA(pb);

	cout << "init flow" << endl;
	initFlow(pb);

	cout << "init second step" << endl;
	nextStep(pb);
	//output_velocity(pb);

	//complex* tv2 = pb.rhs_v;
	complex* tv2 = (complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(tv, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	
	clock_t start_time, end_time;
	double cost;
	double total_cost = 0.0;
	int count = 0;
	for (int i = 0; i < 301; i++) {
		std::cout << "step: " << i << std::endl;
		start_time = clock();
		nextStep(pb);
		//statistics(pb);
		end_time = clock();
		cost = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		total_cost += cost;
		count++;
		std::cout << "time_cost:" << cost << std::endl;
		std::cout << "mean time cost:" << total_cost / count << std::endl;

		if (i % 100 == 0) {
			std::cout << "WRINTING RESULTS TO FILES" << std::endl;
			output_velocity(pb);
		}
	}
	//output_velocity(pb);
	cuCheck(cudaMemcpy(tv2, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");

	return TestSuccess;

	#pragma omp parallel for
	for (int ix = 0; ix < pb.nx / 2 + 1; ix++) {
		for (int iy = 0; iy < pb.ny; iy++) {
			for (int iz = 0; iz < pb.nz; iz++) {
				size_t inc = pb.tPitch / sizeof(complex)
					*((pb.nx / 2 + 1)*iy + ix);
				assert(isEqual(tv[inc].re, tv2[inc].re));
				assert(isEqual(tv[inc].im, tv2[inc].im));
			}
		}
	}

	return TestSuccess;
}
