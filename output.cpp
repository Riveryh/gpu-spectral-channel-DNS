#include "output.h"
#include "transform.cuh"
#include "cuRPCF.h"
#include <string>
#include <fstream>
#include <sstream>
#include "data.h"
#include <iostream>
#include <cstdlib>
#include "velocity.h"


using namespace std;

void write_velocity(const char* filename, real* u, real* v, real* w,
	size_t pitch, int px, int py, int pz);

void output_velocity(problem & pb)
{
	int dim[3] = { pb.mx,pb.my,pb.mz };
	int tDim[3] = { pb.mz,pb.mx,pb.my };
	transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);

	pb.hptr_u = (real*)malloc(pb.pSize);
	pb.hptr_v = (real*)malloc(pb.pSize);
	pb.hptr_w = (real*)malloc(pb.pSize);
	pb.hptr_omega_x = (real*)malloc(pb.pSize);
	pb.hptr_omega_y = (real*)malloc(pb.pSize);
	pb.hptr_omega_z = (real*)malloc(pb.pSize);
	cuCheck(cudaMemcpy(pb.hptr_u, pb.dptr_u.ptr, pb.pSize, cudaMemcpyDeviceToHost),"memcpy");
	cuCheck(cudaMemcpy(pb.hptr_v, pb.dptr_v.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_w, pb.dptr_w.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_x, pb.dptr_omega_x.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_y, pb.dptr_omega_y.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_z, pb.dptr_omega_z.ptr, pb.pSize, cudaMemcpyDeviceToHost), "memcpy");

	ostringstream ss;
	ss << pb.currenStep;
	string velocity_filename = pb.para.ioPara.output_file_prefix + "_velocity." + ss.str() + ".dat";
	string vortricity_filename = pb.para.ioPara.output_file_prefix + "_vortricity." + ss.str() + ".dat";
	cout << "writing results to " << velocity_filename << " and " << vortricity_filename << endl;
	write_velocity(velocity_filename.c_str(), pb.hptr_u, pb.hptr_v, pb.hptr_w,
		pb.pitch, pb.px, pb.py, pb.pz); 
	write_velocity(vortricity_filename.c_str(), pb.hptr_omega_x, pb.hptr_omega_y, pb.hptr_omega_z,
			pb.pitch, pb.px, pb.py, pb.pz);

	safeFree(pb.hptr_u);
	safeFree(pb.hptr_v);
	safeFree(pb.hptr_w);
	safeFree(pb.hptr_omega_x);
	safeFree(pb.hptr_omega_y);
	safeFree(pb.hptr_omega_z);
	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
}

void write_velocity(const char* filename, real* u, real* v, real* w,
	size_t pitch, int px,int py, int pz) {
	ofstream outfile;
	outfile.open(filename, ios::binary);
	for(int k=0;k<pz;k++)	{
		for (int j = 0; j < py; j++) {
			//for (int i = 0; i < px; i++) {
				size_t inc = pitch*(py*k + j) / sizeof(real);
				outfile.write((char*)(u + inc), sizeof(real)*px);
			//}
		}
	}
	for (int k = 0; k<pz; k++) {
		for (int j = 0; j < py; j++) {
			//for (int i = 0; i < px; i++) {
			size_t inc = pitch*(py*k + j) / sizeof(real);
			outfile.write((char*)(v + inc), sizeof(real)*px);
			//}
		}
	}
	for (int k = 0; k<pz; k++) {
		for (int j = 0; j < py; j++) {
			//for (int i = 0; i < px; i++) {
			size_t inc = pitch*(py*k + j) / sizeof(real);
			outfile.write((char*)(w + inc), sizeof(real)*px);
			//}
		}
	}	
	outfile.close();
}

void write_recover_data(problem& pb, char* filename) {
	ofstream outfile;
	outfile.open(filename, ios::binary);
	if (!outfile.is_open()) {
		cerr << "cannot open recovery file" << endl;
		exit(0);
	}

	outfile.write((char*)&pb.mx, sizeof(int));
	outfile.write((char*)&pb.my, sizeof(int));
	outfile.write((char*)&pb.mz, sizeof(int));

	outfile.write((char*)&pb.pSize, sizeof(size_t));
	outfile.write((char*)&pb.tSize, sizeof(size_t));

	outfile.write((char*)pb.lambx0, sizeof(complex)*pb.nz);
	outfile.write((char*)pb.lambz0, sizeof(complex)*pb.nz);
	outfile.write((char*)pb.lambx0_p, sizeof(complex)*pb.nz);
	outfile.write((char*)pb.lambz0_p, sizeof(complex)*pb.nz);
	outfile.write((char*)pb.tv0, sizeof(complex)*pb.nz);
	outfile.write((char*)pb.tomega_y_0, sizeof(complex)*pb.nz);

	outfile.write((char*)pb.rhs_v, pb.tSize);
	outfile.write((char*)pb.rhs_v_p, pb.tSize);
	outfile.write((char*)pb.rhs_omega_y, pb.tSize);

	outfile.write((char*)pb.nonlinear_v, pb.tSize);
	//	outfile.write((char*)pb.nonlinear_v_p, pb.tSize);
	outfile.write((char*)pb.nonlinear_omega_y, pb.tSize);
	//	outfile.write((char*)pb.nonlinear_omega_y_p, pb.tSize);
}

#define READ_AND_CHECK(infile, buffer, v, size);\
	infile.read((char*)&buffer,size);\
	if(buffer!=v){cerr<<"wrong parameter"<<endl;exit(0);}

void read_recover_data(problem& pb, char* filename) {
	ifstream outfile;
	outfile.open(filename, ios::binary);
	if (!outfile.is_open()) {
		cerr << "cannot open recovery file" << endl;
		exit(0);
	}

	int mx, my, mz;
	size_t pSize, tSize;

	READ_AND_CHECK(outfile, mx, pb.mx, sizeof(int));
	READ_AND_CHECK(outfile, my, pb.my, sizeof(int));
	READ_AND_CHECK(outfile, mz, pb.mz, sizeof(int));
	READ_AND_CHECK(outfile, pSize, pb.pSize, sizeof(size_t));
	READ_AND_CHECK(outfile, tSize, pb.tSize, sizeof(size_t));

	outfile.read((char*)pb.lambx0, sizeof(complex)*pb.nz);
	outfile.read((char*)pb.lambz0, sizeof(complex)*pb.nz);
	outfile.read((char*)pb.lambx0_p, sizeof(complex)*pb.nz);
	outfile.read((char*)pb.lambz0_p, sizeof(complex)*pb.nz);
	outfile.read((char*)pb.tv0, sizeof(complex)*pb.nz);
	outfile.read((char*)pb.tomega_y_0, sizeof(complex)*pb.nz);

	outfile.read((char*)pb.rhs_v, pb.tSize);
	outfile.read((char*)pb.rhs_v_p, pb.tSize);
	outfile.read((char*)pb.rhs_omega_y, pb.tSize);

	outfile.read((char*)pb.nonlinear_v, pb.tSize);
	//	outfile.read((char*)pb.nonlinear_v_p, pb.tSize);
	outfile.read((char*)pb.nonlinear_omega_y, pb.tSize);
	//	outfile.read((char*)pb.nonlinear_omega_y_p, pb.tSize);

	// copy velocity and votricity to GPU and compute other components.
	cuCheck(cudaMemcpy(pb.dptr_tw.ptr, pb.rhs_v, pb.tSize, cudaMemcpyHostToDevice),"cpy");
	cuCheck(cudaMemcpy(pb.dptr_tomega_z.ptr, pb.rhs_omega_y, pb.tSize, cudaMemcpyHostToDevice), "cpy");
	getUVW(pb);
}