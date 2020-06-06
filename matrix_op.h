#pragma once
#include "data.h"

int inverse(cuRPCF::complex* mat, int N);
int m_multi_v(cuRPCF::complex* mat, cuRPCF::complex* v, const int N);