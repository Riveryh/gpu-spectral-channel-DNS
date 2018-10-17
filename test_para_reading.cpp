#include "test_para_reading.h"
#include "parameters.h"
#include <iostream>
using namespace std;

TestResult test_read_para() {
	RPCF_Paras para("parameter.txt");
	cout << para.numPara.mx << endl;
	return TestSuccess;
}