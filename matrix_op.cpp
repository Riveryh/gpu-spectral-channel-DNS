#include "mkl.h"
#include "matrix_op.h"

#ifdef REAL_DOUBLE
#define MKL_FACTOR zgetrf
#define MKL_INV zgetri
#define MKL_COMPLEX MKL_Complex16
#else
#define MKL_FACTOR cgetrf
#define MKL_INV cgetri
#define MKL_COMPLEX MKL_Complex8
#endif

int inverse(complex* mat, const int N) {
	int* ipiv = (int*)malloc((N) * sizeof(int));
	const int lwork = (N) * 64 * sizeof(MKL_COMPLEX);
	MKL_COMPLEX* work = (MKL_COMPLEX*)malloc(lwork);
	int info;
	MKL_FACTOR(&N, &N, (MKL_COMPLEX*)mat, &N, ipiv, &info);
	if (info != 0) {
		return info;
	}
	MKL_INV(&N, (MKL_COMPLEX*)mat, &N, ipiv, work, &lwork, &info);
	if (info != 0) {
		return info;
	}
	free(ipiv);
	free(work);
	return 0;
}

#ifdef REAL_DOUBLE
#define CBLAS_CGEMV cblas_zgemv
#else
#define CBLAS_CGEMV cblas_cgemv
#endif

// using intel library
int m_multi_v_cblas(complex* mat, complex* v, const int N) {
	complex buffer[MAX_NZ];
	complex alpha = complex(1.0,0.0);
	complex beta = complex(0.0, 0.0);
	CBLAS_CGEMV(CblasRowMajor, CblasNoTrans, N, N, &alpha, mat, N, v, 1, &beta, buffer, 1);
	for (int i = 0; i < N; i++) {
		v[i] = buffer[i];
	}
	return 0;
}

int m_multi_v_myversion(complex* mat, complex* v, const int N) {
	complex temp[MAX_NZ];
	complex v_cache[MAX_NZ];
	for (int i = 0; i < N; i++) {
		v_cache[i] = v[i];
	}
	//complex* temp = (complex*)malloc(N*sizeof(complex));
	for (int i = 0; i < N; i++) {
		temp[i] = 0.0;
		complex* row = mat + N*i;
		for (int j = 0; j < N; j++) {
			temp[i] = temp[i] + row[j] * v_cache[j];
		}
	}
	for (int i = 0; i < N; i++) {
		v[i] = temp[i];
	}
	//free(temp);
	return 0;
}

int m_multi_v(complex* mat, complex* v, const int N) {
	//return m_multi_v_myversion(mat, v, N);
	return m_multi_v_cblas(mat,v,N);
}