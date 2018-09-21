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
	MKL_FACTOR(&N, &N, (MKL_Complex16*)mat, &N, ipiv, &info);
	if (info != 0) {
		return info;
	}
	MKL_INV(&N, (MKL_Complex16*)mat, &N, ipiv, work, &lwork, &info);
	if (info != 0) {
		return info;
	}
	free(ipiv);
	free(work);
	return 0;
}

int m_multi_v(complex* mat, complex* v, const int N) {

	complex temp[MAX_NZ];
	//complex* temp = (complex*)malloc(N*sizeof(complex));
	for (int i = 0; i < N; i++) {
		temp[i] = 0.0;
		for (int j = 0; j < N; j++) {
			size_t inc = N*i + j;
			temp[i] = temp[i] + mat[inc] * v[j];
		}
	}
	for (int i = 0; i < N; i++) {
		v[i] = temp[i];
	}
	//free(temp);
	return 0;
}