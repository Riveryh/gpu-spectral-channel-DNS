#include "rhs.cuh"
#include "nonlinear.cuh"
#include "linear.h"
#include "cuRPCF.h"
#include "transform.cuh"
void save_zero_wave_number_lamb(problem& pb);

__host__ int get_rhs_v(problem& pb) {
	transform(BACKWARD, pb);
	getNonlinear(pb);
	
	// transform the nonlinear term into physical space.
	cheby_s2p(pb.dptr_tLamb_x, pb.mx/2+1, pb.my, pb.mz, No_Padding);
	cheby_s2p(pb.dptr_tLamb_y, pb.mx/2+1, pb.my, pb.mz, No_Padding);

	//save previous step
	swap(pb.nonlinear_omega_y, pb.nonlinear_omega_y_p);
	swap(pb.nonlinear_v, pb.nonlinear_v_p);
	size_t tsize = pb.tSize;// pb.tPitch * (pb.mx / 2 + 1) * pb.my;
	cuCheck(cudaMemcpy(pb.nonlinear_v, pb.dptr_tLamb_x.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.nonlinear_omega_y, pb.dptr_tLamb_y.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	//TODO: NONLINEAR TIME SCHEME
	save_zero_wave_number_lamb(pb);
	get_linear_v(pb);
	return 0;

}

__host__ int get_rhs_omega(problem& pb) {
	return get_linear_omega_y(pb);
	safeCudaFree(pb.dptr_tLamb_x.ptr);
	safeCudaFree(pb.dptr_tLamb_y.ptr);
	safeCudaFree(pb.dptr_tLamb_z.ptr);
}

void save_zero_wave_number_lamb(problem& pb) {
	swap(pb.lambx0, pb.lambx0_p);
	swap(pb.lambz0, pb.lambz0_p);
	for (int i = 0; i < pb.nz; i++) {
		pb.lambx0[i] = pb.nonlinear_v[i];
		pb.lambz0[i] = pb.nonlinear_omega_y[i];
	}
}
