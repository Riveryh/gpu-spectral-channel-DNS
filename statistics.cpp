#include "statistics.h"
#define PITCHED_LOOP_BEGIN(mx,my,mz) \
	for(int loop_index_i=0;loop_index_i<mx;loop_index_i++){\
	for(int loop_index_j=0;loop_index_j<my;loop_index_j++){\
	for(int loop_index_k=0;loop_index_k<mz;loop_index_k++){
#define PITCHED_LOOP_END(); }}}
#define PITCHED_INCREMENT(npitch) \
	npitch * (my*loop_index_k + loop_index_j) + loop_index_i
#include "transform.cuh"
#include "cuRPCF.h"
#include <iostream>
using namespace std;

real statistic_max(real* u, size_t pitch, int mx, int my, int mz) {
	real max = 0.0;
	PITCHED_LOOP_BEGIN(mx, my, mz)
		size_t inc = PITCHED_INCREMENT(pitch / sizeof(real));
		if (max < abs(u[inc])) {
			max = abs(u[inc]);
		}
	PITCHED_LOOP_END();
	return max;
}

void statistics(problem& pb) {
	int dim[3] = { pb.mx,pb.my,pb.mz };
	int tDim[3] = { pb.mz,pb.mx,pb.my };
	transform_3d_one(BACKWARD, pb.dptr_u, pb.dptr_tu, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_v, pb.dptr_tv, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_w, pb.dptr_tw, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_x, pb.dptr_tomega_x, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_y, pb.dptr_tomega_y, dim, tDim);
	transform_3d_one(BACKWARD, pb.dptr_omega_z, pb.dptr_tomega_z, dim, tDim);

	pb.hptr_u = (real*)malloc(pb.size);
	pb.hptr_v = (real*)malloc(pb.size);
	pb.hptr_w = (real*)malloc(pb.size);
	pb.hptr_omega_x = (real*)malloc(pb.size);
	pb.hptr_omega_y = (real*)malloc(pb.size);
	pb.hptr_omega_z = (real*)malloc(pb.size);
	cuCheck(cudaMemcpy(pb.hptr_u, pb.dptr_u.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_v, pb.dptr_v.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_w, pb.dptr_w.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_x, pb.dptr_omega_x.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_y, pb.dptr_omega_y.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.hptr_omega_z, pb.dptr_omega_z.ptr, pb.size, cudaMemcpyDeviceToHost), "memcpy");

#define MAX_FILED_NUMBER(u) statistic_max(u,pb.pitch,pb.mx,pb.my,pb.pz)

	cout << "max u:" << MAX_FILED_NUMBER(pb.hptr_u) << endl;
	cout << "max v:" << MAX_FILED_NUMBER(pb.hptr_v) << endl;
	cout << "max w:" << MAX_FILED_NUMBER(pb.hptr_w) << endl;
	cout << "max ox:" << MAX_FILED_NUMBER(pb.hptr_omega_x) << endl;
	cout << "max oy:" << MAX_FILED_NUMBER(pb.hptr_omega_y) << endl;
	cout << "max oz:" << MAX_FILED_NUMBER(pb.hptr_omega_z) << endl;

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