#pragma once

#include "sys/types.h"
#include "cudadefs.h"

namespace gpu {

double update_wrapper(double *A, size_t NX, size_t NY, size_t NZ, dim3 blocks_per_grid, dim3 threads_per_block, double* eps_out);

} // gpu
