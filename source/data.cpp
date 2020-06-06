#include "../include/data.h"

void problem::initVars() {
	const REAL PI = 4.0*atan(1.0);
	lx = 2 * PI*aphi;
	ly = 2 * PI*beta;
	hptr_omega_z = nullptr;
	nx = mx / 3 * 2;
	ny = my / 3 * 2;
	nz = mz / 4 + 1;
	px = mx;
	py = my;
	pz = mz / 2 + 1;

	// compute cuda dimensions.
	nThread.x = 4;
	nThread.y = 4;
	nDim.x = my / nThread.x;
	nDim.y = mz / nThread.y;
	if (my%nThread.x != 0) nDim.x++;
	if (mz%nThread.y != 0) nDim.y++;

	ntDim.x = (nx / 2 + 1) / nThread.x;
	ntDim.y = ny / nThread.y;
	if ((nx / 2 + 1) % nThread.x != 0) ntDim.x++;
	if (ny % nThread.y != 0) ntDim.y++;

	npDim.x = py / nThread.x;
	npDim.y = pz / nThread.y;
	if (py%nThread.x != 0) npDim.x++;
	if (pz%nThread.y != 0) npDim.y++;
}