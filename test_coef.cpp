#include "test_coef.h"
#include "coefficient.cuh"
#include <cassert>
TestResult test_get_T_matrix()
{
	real *T0, *T2, *T4;
	int N = 100;
	T0 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	T2 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	T4 = (real*)malloc((N + 1)*(N + 1) * sizeof(real));
	get_T_matrix(N, T0, T2, T4);
	size_t inc1 = (N + 1) * 16 + 100;
	size_t inc2 = (N + 1) * 42 + 95;
	size_t inc3 = (N + 1) * 0 + 4;
	size_t inc4 = (N + 1) * 100 + 5;
	real PRECISION = 1e-6;
	assert(isEqual(T4[inc1] / 1e9, 1.846564249235145, PRECISION));
	assert(isEqual(T4[inc2] / 1e7, 8.842927487588026, PRECISION));
	assert(isEqual(T4[inc3], 192.0, PRECISION));
	assert(isEqual(T4[inc4], -1920.0, PRECISION));
	for (int i = 0; i <= N; i++) {
		for (int j = 0; j <= 3; j++) {
			size_t inc_j = i*(N + 1) + j;
			assert(isEqual(T4[inc_j],0.0,PRECISION));
		}
	}
	return TestSuccess;
}
