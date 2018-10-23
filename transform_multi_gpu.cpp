#include "transform.cuh"
#include "transform_multi_gpu.h"
#include "cuRPCF.h"
#include <cassert>
#include "pthread.h"

cufftHandle planXYr2c_M[NUM_GPU], planXYc2r_M[NUM_GPU], planZ_pad_M[NUM_GPU];
int dev_id[NUM_GPU];
dim3 __dim;
cudaPitchedPtr peer_tPtr[3];
cudaPitchedPtr peer_Ptr[3];
int peer_id = 1;

cudaStream_t peerCpyStream;

int init_FFT_MGPU(problem& pb) {
	cufftResult res;
	const int mx = pb.mx;
	const int my = pb.my;
	const int mz = pb.mz;
	const int inPitch = pb.pitch;
	const int outPitch = pb.tPitch;
	const int pmx = inPitch / sizeof(real);
	const int pmz = outPitch / sizeof(complex);
	const int nx = mx / 3 * 2;
	const int ny = my / 3 * 2;

	const int istride = 1;
	int inembed[2] = { my, pmx };
	int idist = pmx*my;

	int inembed2[2] = { my,pmx / 2 };
	int idist2 = pmx / 2 * my;
	int dim2[2] = { my,mx };

	int dim1[1] = { mz };
	int onembed[1] = { pmz };
	const int odist = pmz;
	const int ostride = 1;

	int dim1_no_pad[1] = { mz / 2 };


	int ndev;
	cudaGetDeviceCount(&ndev);
	int max_gpu = 3; NUM_GPU < ndev ? NUM_GPU : ndev;
	for (int i = 1; i < max_gpu; i++) {
		cudaSetDevice(i);
		res = cufftPlanMany(&planXYr2c_M[i], 2, dim2, inembed, istride, idist,
			inembed2, istride, idist2, myCUFFT_R2C, pb.pz);
		assert(res == CUFFT_SUCCESS);
		res = cufftPlanMany(&planXYc2r_M[i], 2, dim2, inembed2, istride, idist2,
			inembed, istride, idist, myCUFFT_C2R, pb.pz);
		assert(res == CUFFT_SUCCESS);
		res = cufftPlanMany(&planZ_pad_M[i], 1, dim1, onembed, ostride, odist,
			onembed, ostride, odist, myCUFFT_C2C, (nx / 2 + 1)*ny);
		assert(res == CUFFT_SUCCESS);
	}
	planXYr2c_M[0] = planXYr2c;
	planXYc2r_M[0] = planXYc2r;
	planZ_pad_M[0] = planZ_pad;
	
	cudaError_t err;
	err = cudaStreamCreate(&peerCpyStream);
	assert(err == cudaSuccess);

	__dim = dim3(pb.mx, pb.my, pb.mz);
	return 0;
}

pthread_mutex_t mutex_gpu_thread_start[NUM_GPU];
pthread_cond_t cond_gpu_thread_start[NUM_GPU];
pthread_mutex_t mutex_gpu_thread_end[NUM_GPU];
pthread_cond_t cond_gpu_thread_end[NUM_GPU];

void* start_gpu_transorm_backward_thread(void* data) {
	cudaSetDevice(dev_id[peer_id]);
	pthread_mutex_lock(&mutex_gpu_thread_start[peer_id]);
	while (true) {
		pthread_cond_wait(&cond_gpu_thread_start[peer_id],&mutex_gpu_thread_start[peer_id]);
		cuCheck(cudaDeviceSynchronize(), "wait for current cpy stop");
		for (int i = 0; i < 3; i++)
		{
			transform_one_mGPU(BACKWARD, peer_Ptr[i], peer_tPtr[i], __dim, dev_id[1]);
		}
		cuCheck(cudaDeviceSynchronize(), "wait for current transform stop");
		
		pthread_mutex_lock(&mutex_gpu_thread_end[peer_id]);
		pthread_cond_signal(&cond_gpu_thread_end[peer_id]);
		pthread_mutex_unlock(&mutex_gpu_thread_end[peer_id]);
	}
	pthread_mutex_unlock(&mutex_gpu_thread_start[peer_id]);
}

int init_multi_gpu_thread() {
	for (int i = 0; i < NUM_GPU; i++) {
		pthread_mutex_init(&mutex_gpu_thread_start[i], NULL);
		pthread_cond_init(&cond_gpu_thread_start[i], NULL);
		pthread_mutex_init(&mutex_gpu_thread_end[i], NULL);
		pthread_cond_init(&cond_gpu_thread_end[i], NULL);
	}
	pthread_t pid;
	pthread_create(&pid, NULL, start_gpu_transorm_backward_thread ,NULL);
	return 0;
}

int transform_one_mGPU(DIRECTION dir, cudaPitchedPtr& Ptr, cudaPitchedPtr& tPtr, dim3 dims, int dev_id) {
	int dim[3] = { dims.x,dims.y,dims.z };
	int tDim[3] = { dims.z,dims.x,dims.y };
	cuCheck(cudaSetDevice(dev_id));
	if (dir == FORWARD) {
		cuFFTcheck(CUFFTEXEC_R2C(planXYr2c_M[dev_id], (CUFFTREAL*)Ptr.ptr, (CUFFTCOMPLEX*)Ptr.ptr),"XY forward");
		normalize(Ptr, dims, 1.0 / dims.x / dims.y);
		cuCheck(cudaDeviceSynchronize(),"XY Forward");
		cuCheck(myCudaMalloc(tPtr, ZXY_3D, dev_id), "my cudaMalloc");
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);
		cuCheck(myCudaFree(Ptr, XYZ_3D, dev_id), "my cuda free at transform");
		cheby_p2s(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);
	}
	else if (dir == BACKWARD) {
		cheby_s2p(tPtr, dim[0] / 2 + 1, dim[1], dim[2]);
		cuCheck(myCudaMalloc(Ptr, XYZ_3D, dev_id), "my cudaMalloc");
		cuda_transpose(dir, Ptr, tPtr, dim, tDim);
		cuCheck(myCudaFree(tPtr, ZXY_3D, dev_id), "my cuda free at transform");
		setZeros((complex*)Ptr.ptr, Ptr.pitch, dim3(dim[0], dim[1], dim[2]));
		cuFFTcheck(CUFFTEXEC_C2R(planXYc2r_M[dev_id], (CUFFTCOMPLEX*)Ptr.ptr,(CUFFTREAL*)Ptr.ptr), "XY backward");
	}
	else {
		return -1;
	}
	return 0;
}

int transform_MGPU(problem& pb, DIRECTION direction) {
	if (direction == BACKWARD) {
		// copy data tox,toy,toz to peer device
		cuCheck(myCudaMalloc(peer_tPtr[2], ZXY_3D, dev_id[1]),"malloc peer memory");
		cuCheck(myCudaMalloc(peer_tPtr[1], ZXY_3D, dev_id[1]), "malloc peer memory");
		cuCheck(myCudaMalloc(peer_tPtr[0], ZXY_3D, dev_id[1]), "malloc peer memory");

		cudaMemcpyPeerAsync(peer_tPtr[0].ptr, dev_id[1], pb.dptr_tomega_x.ptr, dev_id[0], pb.tSize, peerCpyStream);
		cudaMemcpyPeerAsync(peer_tPtr[1].ptr, dev_id[1], pb.dptr_tomega_y.ptr, dev_id[0], pb.tSize, peerCpyStream);
		cudaMemcpyPeerAsync(peer_tPtr[2].ptr, dev_id[1], pb.dptr_tomega_z.ptr, dev_id[0], pb.tSize, peerCpyStream);
		
		// launch transform on peer device
		pthread_mutex_lock(&mutex_gpu_thread_start[peer_id]);
		pthread_mutex_lock(&mutex_gpu_thread_end[peer_id]);
		pthread_cond_signal(&cond_gpu_thread_start[peer_id]);
		pthread_mutex_unlock(&mutex_gpu_thread_start[peer_id]);

		// launch transform on host thread
		transform_one_mGPU(direction, pb.dptr_u, pb.dptr_tu, dim3(pb.mx, pb.my, pb.mz), dev_id[0]);
		transform_one_mGPU(direction, pb.dptr_v, pb.dptr_tv, dim3(pb.mx, pb.my, pb.mz), dev_id[0]);
		transform_one_mGPU(direction, pb.dptr_w, pb.dptr_tw, dim3(pb.mx, pb.my, pb.mz), dev_id[0]);

		// wait for peer launch stop
		pthread_cond_wait(&cond_gpu_thread_end[peer_id],&mutex_gpu_thread_end[peer_id]);
		pthread_mutex_unlock(&mutex_gpu_thread_end[peer_id]);

		// alloc host memory for recieving.
		cuCheck(myCudaFree(pb.dptr_tomega_x, XYZ_3D, dev_id[0]));
		cuCheck(myCudaFree(pb.dptr_tomega_y, XYZ_3D, dev_id[0]));
		cuCheck(myCudaFree(pb.dptr_tomega_z, XYZ_3D, dev_id[0]));
		cuCheck(myCudaMalloc(pb.dptr_omega_x, XYZ_3D, dev_id[0]));
		cuCheck(myCudaMalloc(pb.dptr_omega_y, XYZ_3D, dev_id[0]));
		cuCheck(myCudaMalloc(pb.dptr_omega_z, XYZ_3D, dev_id[0]));

		// copy data ox,oy,oz from peer device
		cudaMemcpyPeerAsync(pb.dptr_omega_x.ptr, dev_id[0], peer_Ptr[0].ptr, dev_id[1], pb.pSize, peerCpyStream);
		cudaMemcpyPeerAsync(pb.dptr_omega_y.ptr, dev_id[0], peer_Ptr[1].ptr, dev_id[1], pb.pSize, peerCpyStream);
		cudaMemcpyPeerAsync(pb.dptr_omega_z.ptr, dev_id[0], peer_Ptr[2].ptr, dev_id[1], pb.pSize, peerCpyStream);

	}
	return 0;
}