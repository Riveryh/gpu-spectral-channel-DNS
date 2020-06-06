//#include "mkl.h"
#include "../include/matrix_op.h"

#define EIGEN_USE_MKL_ALL
#include "eigen-3.3.7/Eigen/Dense"
#include <complex>

//
//#ifdef REAL_DOUBLE
//#define MKL_FACTOR zgetrf
//#define MKL_INV zgetri
//#define MKL_COMPLEX MKL_Complex16
//#else
//#define MKL_FACTOR cgetrf
//#define MKL_INV cgetri
//#define MKL_COMPLEX MKL_Complex8
//#endif

typedef Eigen::Matrix<std::complex<REAL>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CoefType;
typedef Eigen::Matrix<std::complex<REAL>, Eigen::Dynamic, 1> VecType;

int inverse(cuRPCF::complex* mat, const int N) {
	Eigen::Map<CoefType> matrix(reinterpret_cast<std::complex<REAL>*>(mat), N, N);
	auto invMat = matrix.inverse();
	matrix = invMat;
	return 0;
}

//#ifdef REAL_DOUBLE
//#define CBLAS_CGEMV cblas_zgemv
//#else
//#define CBLAS_CGEMV cblas_cgemv
//#endif
//
//// using intel library
//int m_multi_v_cblas(cuRPCF::complex* mat, cuRPCF::complex* v, const int N) {
//	cuRPCF::complex buffer[MAX_NZ];
//	cuRPCF::complex alpha = cuRPCF::complex(1.0,0.0);
//	cuRPCF::complex beta = cuRPCF::complex(0.0, 0.0);
//	CBLAS_CGEMV(CblasRowMajor, CblasNoTrans, N, N, &alpha, mat, N, v, 1, &beta, buffer, 1);
//	for (int i = 0; i < N; i++) {
//		v[i] = buffer[i];
//	}
//	return 0;
//}
//
//int m_multi_v_myversion(cuRPCF::complex* mat, cuRPCF::complex* v, const int N) {
//	cuRPCF::complex temp[MAX_NZ];
//	cuRPCF::complex v_cache[MAX_NZ];
//	for (int i = 0; i < N; i++) {
//		v_cache[i] = v[i];
//	}
//	//cuRPCF::complex* temp = (cuRPCF::complex*)malloc(N*sizeof(cuRPCF::complex));
//	for (int i = 0; i < N; i++) {
//		temp[i] = 0.0;
//		cuRPCF::complex* row = mat + N*i;
//		for (int j = 0; j < N; j++) {
//			temp[i] = temp[i] + row[j] * v_cache[j];
//		}
//	}
//	for (int i = 0; i < N; i++) {
//		v[i] = temp[i];
//	}
//	//free(temp);
//	return 0;
//}
//
inline int m_multi_v_eigen(cuRPCF::complex* mat, cuRPCF::complex* v, const int N) {
	Eigen::Map<CoefType> matrix(reinterpret_cast<std::complex<REAL>*>(mat), N, N);
	Eigen::Map<VecType> vec(reinterpret_cast<std::complex<REAL>*>(v), N);
	vec = matrix * vec;
}

int m_multi_v(cuRPCF::complex* mat, cuRPCF::complex* v, const int N) {
	//return m_multi_v_myversion(mat, v, N);
	return m_multi_v_eigen(mat,v,N);
}