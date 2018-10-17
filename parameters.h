#pragma once
#include<string>

struct RPCF_Numerical_Para {
	int mx;
	int my;
	int mz;
	double n_pi_x;
	double n_pi_y;
	double Re;
	double Ro;
	double dt;
};

struct RPCF_Step_Para {
	int start_step;
	int end_step;
	int save_internal;
};

struct RPCF_IO_Para {
	std::string output_file;
	std::string input_file;
};

struct RPCF_Paras {
	RPCF_Numerical_Para numPara;
	RPCF_Step_Para stepPara;
	RPCF_IO_Para ioPara;
	void read_para(char* filename);
	RPCF_Paras() {};
	RPCF_Paras(char* filename) {
		this->read_para(filename);
	}
};