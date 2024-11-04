#pragma once

#include "sys/types.h"

namespace cpu {

double adi3d(double *A, size_t size);

void init(double *A, size_t size);

} // cpu

#include "adi3d.hpp"
