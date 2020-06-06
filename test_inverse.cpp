#include "test_inverse.h"
#include "main_test.h"
#include "matrix_op.h"
#include "data.h"
#include <cassert>
TestResult test_m_multi_v() {
	const int N = 100;
	cuRPCF::complex* mat1 = (cuRPCF::complex*)malloc(N*N * sizeof(cuRPCF::complex));
	cuRPCF::complex* mat2 = (cuRPCF::complex*)malloc(N * sizeof(cuRPCF::complex));
	assert(mat1 != nullptr);
	assert(mat2 != nullptr);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			size_t inc = N*i + j;
			mat1[inc] = (i == j) ? 10.0 : 0.0;
		}
		mat2[i] = cuRPCF::complex((REAL)i, 0.0);
	}
	m_multi_v(mat1, mat2, N);
	for (int i = 0; i < N; i++) {
		assert(isEqual(mat2[i].re, 10*(REAL)i, 1e-8));
		assert(isEqual(mat2[i].im, 0.0, 1e-8));
	}
	return TestSuccess;
}

TestResult test_inverse() {
	const int N = 100;
	cuRPCF::complex* mat1 = (cuRPCF::complex*)malloc(N*N * sizeof(cuRPCF::complex));
	cuRPCF::complex* mat2 = (cuRPCF::complex*)malloc(N*N * sizeof(cuRPCF::complex));
	assert(mat1 != nullptr);
	assert(mat2 != nullptr);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			size_t inc = N*i + j;
			mat1[inc] = (i == j) ? 10.0 : 0.0;
			mat2[inc] = mat1[inc];
		}
	}
	inverse(mat1, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			size_t inc = N*i + j;
			assert(isEqual(mat1[inc].re , (i == j) ? 0.1 : 0.0,1e-8));
		}
	}
	return TestSuccess;
}