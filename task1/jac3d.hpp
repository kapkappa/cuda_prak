#include "jac3d.h"

#include <math.h>

namespace {
static inline size_t get_index(size_t i, size_t j, size_t k, size_t size) {
    return i * size * size + j * size + k;
}
}

namespace cpu {

double get_eps(const double * __restrict__ A, const double * __restrict__ B, size_t size) {
    size_t NX = size, NY = size, NZ = size;
    double eps = 0.0;

    for (size_t i = 1; i < NX-1; i++) {
        for (size_t j = 1; j < NY-1; j++) {
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                double tmp = B[idx] - A[idx];
                eps += tmp * tmp;
            }
        }
    }
    return sqrt(eps);
}

void jac3d(double *A, double *B, size_t size) {
    size_t NX = size, NY = size, NZ = size;
    size_t offset_i = NZ * NY;
    size_t offset_j = NZ;
    size_t offset_k = 1;

    for (size_t i = 1; i < NX-1; i++) {
        for (size_t j = 1; j < NY-1; j++) {
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                B[idx] = (A[idx - offset_k] + A[idx - offset_j] + A[idx - offset_i] +
                          A[idx + offset_k] + A[idx + offset_j] + A[idx + offset_i]) / 6.0;
            }
        }
    }
}

void init(double *A, double *B, size_t size) {
    size_t NX = size, NY = size, NZ = size;

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t k = 0; k < NZ; k++) {
                size_t idx = get_index(i, j, k, size);
                A[idx] = 0;
                if (i == 0 || j == 0 || k == 0 || i == NX-1 || j == NY-1 || k == NZ-1) {
                    B[idx] = 0.0;
                } else {
                    B[idx] = 4.0 + i + j + k;
                }
            }
        }
    }
}

} // gpu
