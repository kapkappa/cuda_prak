#pragma once

#include "sys/types.h"

namespace gpu {

double update_wrapper(double *A, double *B, size_t NX, size_t NY, size_t NZ);

} // gpu
