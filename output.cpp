#include "output.h"
#include "transform.cuh"
#include "cuRPCF.h"
#include <string>
#include <fstream>
using namespace std;

void write_velocity(char* filename, real* u, real* v, real* w,
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

	write_velocity("velocity.dat", pb.hptr_u, pb.hptr_v, pb.hptr_w,
		pb.pitch, pb.px, pb.py, pb.pz); 
	write_velocity("votricity.dat", pb.hptr_omega_x, pb.hptr_omega_y, pb.hptr_omega_z,
			pb.pitch, pb.px, pb.py, pb.pz);

	safeFree(pb.hptr_u);
	safeFree(pb.hptr_v);
	safeFree(pb.hptr_w);
	safeFree(pb.hptr_omega_x);
	safeFree(pb.hptr_omega_y);
	safeFree(pb.hptr_omega_z);
	transform_3d_one(FORWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(FORWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);
}

void write_velocity(char* filename, real* u, real* v, real* w,
	size_t pitch, int px,int py, int pz) {
	ofstream outfile;
	outfile.open(filename, ios::binary);
	for(int k=0;k<pz;k++)	{
		for (int j = 0; j < py; j++) {
			for (int i = 0; i < px; i++) {
				size_t inc = pitch*(py*k + j) / sizeof(real) + i;
				outfile.write((char*)(u + inc), sizeof(real));
				outfile.write((char*)(v + inc), sizeof(real));
				outfile.write((char*)(w + inc), sizeof(real));
			}
		}
	}
}