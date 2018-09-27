#include "rhs.cuh"
#include "nonlinear.cuh"
#include "linear.h"
#include "cuRPCF.h"
#include "transform.cuh"

__host__ int get_rhs_v(problem& pb) {
	transform(BACKWARD, pb);
	getNonlinear(pb);
	
	// transform the nonlinear term into physical space.
	cheby_s2p(pb.dptr_tLamb_x, pb.mx/2+1, pb.my, pb.mz);
	cheby_s2p(pb.dptr_tLamb_z, pb.mx/2+1, pb.my, pb.mz);

	//save previous step
	swap(pb.nonlinear_omega_y, pb.nonlinear_omega_y_p);
	swap(pb.nonlinear_v, pb.nonlinear_v_p);
	size_t tsize = pb.tSize;// pb.tPitch * (pb.mx / 2 + 1) * pb.my;
	cuCheck(cudaMemcpy(pb.nonlinear_v, pb.dptr_tLamb_x.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	cuCheck(cudaMemcpy(pb.nonlinear_omega_y, pb.dptr_tLamb_z.ptr, tsize, cudaMemcpyDeviceToHost), "memcpy");
	get_linear_v(pb);
	return 0;

}

__host__ int get_rhs_omega(problem& pb) {
	return get_linear_omega_y(pb);
	safeCudaFree(pb.dptr_tLamb_x.ptr);
	safeCudaFree(pb.dptr_tLamb_y.ptr);
	safeCudaFree(pb.dptr_tLamb_z.ptr);
}

