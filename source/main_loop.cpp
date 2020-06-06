#include "../include/cuRPCF.h"
#include "../include/solver.h"
#include "../include/output.h"
#include <cassert>
#include "../include/statistics.h"
#include <time.h>  
#include <stdio.h>  
#include <iostream>
using namespace std;

void run_simulation();

int main() {
	run_simulation();
	//main_test();
	//return main_test();
}

void run_simulation() {
	RPCF_Paras paras("parameter.txt");
	//paras.numPara.dt = 0.0;
	problem pb(paras);
	cout << "init cuda" << endl;
	getDeviceInfo(pb);
	allocDeviceMem(pb);
	cout << "init fft" << endl;
	initFFT(pb);
	cout << "init solver" << endl;
	allocHostMemory(pb);
	initAuxMatrix(pb);
	cout << "init flow" << endl;

	if (pb.para.stepPara.start_type == 0) {

		initFlow(pb);

		cout << "output flow" << endl;
		pb.currenStep = 0;
		output_velocity(pb);
		//output_velocity(pb);

		cuRPCF::complex* tv = (cuRPCF::complex*)malloc(pb.tSize);


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

		allocDeviceMem(pb);

		cout << "init flow" << endl;
		initFlow(pb);

		cout << "init second step" << endl;
		nextStep(pb);
		//output_velocity(pb);
	}
	else if (pb.para.stepPara.start_type == 1) {
		myCudaFree(pb.dptr_omega_z, XYZ_3D); myCudaMalloc(pb.dptr_tomega_z, ZXY_3D);
		myCudaFree(pb.dptr_omega_y, XYZ_3D); myCudaMalloc(pb.dptr_tomega_y, ZXY_3D);
		myCudaFree(pb.dptr_omega_x, XYZ_3D); myCudaMalloc(pb.dptr_tomega_x, ZXY_3D);
		myCudaFree(pb.dptr_w, XYZ_3D); myCudaMalloc(pb.dptr_tw, ZXY_3D);
		myCudaFree(pb.dptr_v, XYZ_3D); myCudaMalloc(pb.dptr_tv, ZXY_3D);
		myCudaFree(pb.dptr_u, XYZ_3D); myCudaMalloc(pb.dptr_tu, ZXY_3D);
		read_recover_data(pb);
	}
	else 
	{
		cerr << "wrong start type" << endl;
		exit(-1);
	}

	//cuRPCF::complex* tv2 = pb.rhs_v;
	cuRPCF::complex* tv2 = (cuRPCF::complex*)malloc(pb.tSize);
	cuRPCF::complex* tv = (cuRPCF::complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(tv, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	
	//clock_t start_time, end_time;
	cudaEvent_t start_trans, end_trans;
	cudaEventCreate(&start_trans);
	cudaEventCreate(&end_trans);

	double cost;
	double total_cost = 0.0;
	float time;
	int count = 0;
	pb.currenStep = pb.para.stepPara.start_step;
	for (int i = pb.para.stepPara.start_step;
		i <= pb.para.stepPara.end_step; i++) 
	{
		std::cout << "step: " << i << std::endl;
		//start_time = clock();

		cudaEventRecord(start_trans);

		nextStep(pb);
		//statistics(pb);
		//end_time = clock();

		cudaEventRecord(end_trans);
		cudaEventSynchronize(end_trans);
		cudaEventElapsedTime(&time, start_trans, end_trans);

		//cost = (double)(end_time - start_time) / CLOCKS_PER_SEC;
		total_cost += time;
		count++;
		std::cout << "time_cost:" << time / 1000.0 << std::endl;
		std::cout << "mean time cost:" << total_cost / count / 1000.0 << std::endl;
		std::cout << "velocity(0,0,0):" << (pb.rhs_v->re) << " " << (pb.rhs_v->im) << std::endl;

		if (i % pb.para.stepPara.save_internal == 0) {
		//if (i > 320) {
			std::cout << "[OUTPUT] WRINTING RESULTS" << std::endl;
			output_velocity(pb);
		}
		if (i % pb.para.stepPara.save_recovery_internal == 0 && i>0) {
			std::cout << "[OUTPUT] WRINTING RECOVERY FILES" << std::endl;
			write_recover_data(pb);
		}
		//if (i == 340) exit(0);
		pb.currenStep++;
	}
}
