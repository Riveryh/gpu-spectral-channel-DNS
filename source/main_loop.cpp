#include "../include/cuRPCF.h"
#include "../include/solver.h"
#include "../include/output.h"
#include "../include/statistics.h"
#include <iostream>

void run_simulation();

int main() {
	run_simulation();
	//main_test();
	//return main_test();
}

void run_simulation() {
	std::string parameter_filename = "../input/parameter.txt";
	std::cout << "Reading settings from " << parameter_filename << std::endl;
	RPCF_Paras paras(parameter_filename);
	//paras.numPara.dt = 0.0;
	problem pb(paras);
	std::cout << "init cuda" << std::endl;
	getDeviceInfo(pb);
	allocDeviceMem(pb);
	std::cout << "init fft" << std::endl;
	initFFT(pb);
	std::cout << "init solver" << std::endl;
	allocHostMemory(pb);
	initAuxMatrix(pb);

	if (pb.para.stepPara.start_type == 0) {

		std::cout << "init flow" << std::endl;
		initFlow(pb);

		std::cout << "output initial flow field" << std::endl;
		pb.currenStep = 0;
		output_velocity(pb);

		std::cout << "extra previous step" << std::endl;
		//compute extra step to get "previous" step data for the actual first step
		nextStep(pb);

		//re-allocate device memory
		destroyMyCudaMalloc();
		allocDeviceMem(pb);

		//std::cout << "init flow" << std::endl;
		initFlow(pb);

		//real first step
		std::cout << "first step" << std::endl;
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
		std::cerr << "wrong start type" << std::endl;
		exit(-1);
	}

	//cuRPCF::complex* tv2 = pb.rhs_v;
	cuRPCF::complex* tv2 = (cuRPCF::complex*)malloc(pb.tSize);
	cuRPCF::complex* tv = (cuRPCF::complex*)malloc(pb.tSize);
	cuCheck(cudaMemcpy(tv, pb.dptr_tomega_y.ptr, pb.tSize, cudaMemcpyDeviceToHost), "cpy");
	
	//clock_t start_time, end_time;
	cudaEvent_t step_start_event, step_end_event;
	cudaEventCreate(&step_start_event);
	cudaEventCreate(&step_end_event);

	double cost;
	double total_cost = 0.0;
	float time;
	int count = 0;
	pb.currenStep = pb.para.stepPara.start_step;
	for (int i = pb.para.stepPara.start_step;
		i <= pb.para.stepPara.end_step; i++) 
	{
		std::cout << std::endl << "STEP: " << i << std::endl;

		cudaEventRecord(step_start_event);

		nextStep(pb);
		//statistics(pb);

		cudaEventRecord(step_end_event);
		cudaEventSynchronize(step_end_event);
		cudaEventElapsedTime(&time, step_start_event, step_end_event);

		total_cost += time;
		count++;
		std::cout << "Time cost of current step:" << time / 1000.0 << std::endl;
		std::cout << "Mean time cost per step:" << total_cost / count / 1000.0 << std::endl;
		std::cout << "velocity(0,0,0):" << (pb.rhs_v->re) << " " << (pb.rhs_v->im) << std::endl;

		if (i % pb.para.stepPara.save_internal == 0) {
			std::cout << "[OUTPUT] WRINTING RESULTS" << std::endl;
			output_velocity(pb);
		}
		if (i % pb.para.stepPara.save_recovery_internal == 0 && i>0) {
			std::cout << "[OUTPUT] WRINTING RECOVERY FILES" << std::endl;
			write_recover_data(pb);
		}
		pb.currenStep++;
	}
}
