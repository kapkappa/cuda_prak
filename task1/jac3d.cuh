#pragma once

#include "sys/types.h"

namespace gpu {

double get_eps_wrapper(double* A, double* B, size_t NX, size_t NY, size_t NZ, double *eps_out);

void update_wrapper(double *A, double *B, size_t NX, size_t NY, size_t NZ);

} // gpu
