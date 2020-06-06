#pragma once
#include "data.h"
void output_velocity(problem& pb);

void write_recover_data(problem& pb, char* filename = "recovery.dat");
void read_recover_data(problem& pb, char* filename = "recovery.dat");
