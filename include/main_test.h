#pragma once
#include <iostream>

typedef enum {
	TestSuccess,
	TestFailed
} TestResult;

TestResult main_test();

void RUN_TEST(TestResult func, std::string s);
