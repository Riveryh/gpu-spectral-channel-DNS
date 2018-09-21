#include "test_inverse.h"
#include "main_test.h"
#include "matrix_op.h"
#include "data.h"
#include <cassert>
TestResult test_m_multi_v() {
	const int N = 100;
	complex* mat1 = (complex*)malloc(N*N * sizeof(complex));
	complex* mat2 = (complex*)malloc(N * sizeof(complex));
	assert(mat1 != nullptr);
	assert(mat2 != nullptr);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			size_t inc = N*i + j;
			mat1[inc] = (i == j) ? 10.0 : 0.0;
		}
		mat2[i] = complex((real)i, 0.0);
	}
	m_multi_v(mat1, mat2, N);
	for (int i = 0; i < N; i++) {
		assert(isEqual(mat2[i].re, 10*(real)i, 1e-8));
		assert(isEqual(mat2[i].im, 0.0, 1e-8));
	}
	return TestSuccess;
}

TestResult test_inverse() {
	const int N = 100;
	complex* mat1 = (complex*)malloc(N*N * sizeof(complex));
	complex* mat2 = (complex*)malloc(N*N * sizeof(complex));
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