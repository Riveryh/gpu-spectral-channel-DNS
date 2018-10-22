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
	int start_type;	// 0 for new, 1 for recovery.
	int start_step;
	int end_step;
	int save_internal;
	int save_recovery_internal;
};

struct RPCF_IO_Para {
	std::string output_file_prefix;
	std::string recovery_file_prefix;
};

struct RPCF_Paras {
	RPCF_Numerical_Para numPara;
	RPCF_Step_Para stepPara;
	RPCF_IO_Para ioPara;
	void read_para(std::string filename);
	RPCF_Paras() {};
	RPCF_Paras(std::string filename) {
		this->read_para(filename);
	}
};