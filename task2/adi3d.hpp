#include "adi3d.h"

#include <math.h>

namespace {
static inline size_t get_index(size_t i, size_t j, size_t k, size_t size) {
    return i * size * size + j * size + k;
}
}

namespace cpu {

double adi3d(double *A, size_t size) {
    size_t NX = size, NY = size, NZ = size;
    size_t offset_i = NZ * NY;
    size_t offset_j = NZ;
    size_t offset_k = 1;

    double eps = 0.0;

    for (size_t i = 1; i < NX-1; i++)
        for (size_t j = 1; j < NY-1; j++)
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                A[idx] = (A[idx - offset_i] + A[idx + offset_i]) / 2.0;
            }

    for (size_t i = 1; i < NX-1; i++)
        for (size_t j = 1; j < NY-1; j++)
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                A[idx] = (A[idx - offset_j] + A[idx + offset_j]) / 2.0;
            }

    for (size_t i = 1; i < NX-1; i++)
        for (size_t j = 1; j < NY-1; j++)
            for (size_t k = 1; k < NZ-1; k++)
            {
                size_t idx = get_index(i, j, k, size);
                double tmp1 = (A[idx - offset_k] + A[idx + offset_k]) / 2.0;
                double tmp2 = fabs(A[idx] - tmp1);
                eps = std::max(eps, tmp2);
                A[idx] = tmp1;
            }

    return eps;
}

void init(double *A, size_t size) {
    size_t NX = size, NY = size, NZ = size;

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t k = 0; k < NZ; k++) {
                size_t idx = get_index(i, j, k, size);
                if (i == 0 || j == 0 || k == 0 || i == NX-1 || j == NY-1 || k == NZ-1) {
                    A[idx] = 10.0 * i / (NX - 1) + 10.0 * j / (NY - 1) + 10.0 * k / (NZ - 1);
                } else {
                    A[idx] = 0.0;
                }
            }
        }
    }
}

} // cpu
