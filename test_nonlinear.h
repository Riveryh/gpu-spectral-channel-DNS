#pragma once
#include "main_test.h"
#include "data.h"
TestResult test_nonlinear();
void setFlow(problem& pb);

void setFlow_basic3(problem& pb);

TestResult check_lamb(problem& pb);

TestResult check_lamb_basic3(problem& pb);
TestResult check_nonlinear_basic(problem& pb);
void set_lamb_nonlinear_basic(problem& pb);

TestResult check_nonlinear_basic2(problem& pb);
TestResult check_nonlinear_basic3(problem& pb);
void set_lamb_nonlinear_basic2(problem& pb);

void set_lamb_nonlinear_complex(problem& pb);
TestResult check_nonlinear_complex(problem& pb);