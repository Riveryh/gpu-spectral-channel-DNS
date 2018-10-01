#include "main_test.h"
#include "test_transform.h"
#include "test_nonlinear.h"
#include "test_coef.h"
#include "test_inverse.h"
#include "test_para_reading.h"
#include "test_getUVW.h"
#include "test_rhs.h"
#include "test_dt_0.h"
#include <iostream>
using namespace std;
#include <cstdio>

int main() {
	return main_test();
}

TestResult main_test(){
	//RUN_TEST(test_transform(),"test_transform");
	//RUN_TEST(test_nonlinear(), "test nonlinear");
	//RUN_TEST(test_get_T_matrix(), "test chebyshev");
	//RUN_TEST(test_m_multi_v(), "test matrix inversion");
	//RUN_TEST(test_inverse(), "test matrix inversion");
	//RUN_TEST(test_read_para(), "test parameter reading");
	//RUN_TEST(Test_coef_v(), "Test coef v");
	//RUN_TEST(Test_coef_omega(), "Test coef omega");
	//RUN_TEST(test_getUVW(), "Test UVW");
	//RUN_TEST(test_rhs(), "Test rhs");
	RUN_TEST(test_dt_0(), "Test equality of dt = 0");
	return TestSuccess;
}

void RUN_TEST(TestResult func, string s) {
	if (func == TestSuccess) {
		printf("[TEST SUCCESS]: %s\n", s.c_str());
		return;
	}
	else
	{
		char c;
		printf("%s\n", s.c_str());
		cin >> c;
		exit(func);
	}
}