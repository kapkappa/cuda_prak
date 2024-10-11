#pragma once

#include "sys/types.h"

namespace cpu {

double get_eps(const double * __restrict__ A, const double * __restrict__ B, size_t size);

void jac3d(double *A, double *B, size_t size);

void init(double *A, double *B, size_t size);

} // cpu

#include "jac3d.hpp"
