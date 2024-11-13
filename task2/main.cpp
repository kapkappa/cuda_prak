/* ADI program */

#include <string>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "adi3d.h"
#include "adi3d.cuh"
#include "cudadefs.h"

#define DIFF 1E-6
#define EPS 0.07249074

namespace {
static inline double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}
}

int main(int argc, char **argv) {

    // Default values of input parameters
    int argc_indx = 0;
    int iters = 100;
    size_t size = 384;
    double max_eps = 0.01;
    std::string driver = "CPU";
    enum class driver_t {CPU, GPU} drv = driver_t::CPU;

    // Parsing input parameters
    while (argc_indx < argc) {
        if (!strcmp(argv[argc_indx], "-size")) {
            argc_indx++;
            size = atoi(argv[argc_indx]);
        } else if (!strcmp(argv[argc_indx], "-iters")) {
            argc_indx++;
            iters = atoi(argv[argc_indx]);
        } else if (!strcmp(argv[argc_indx], "-driver")) {
            argc_indx++;
            if (!strcmp(argv[argc_indx], "GPU")) {
                drv = driver_t::GPU;
                driver = "GPU";
            } else if (!strcmp(argv[argc_indx], "CPU")) {
                drv = driver_t::CPU;
                driver = "CPU";
            } else {
                printf("Wrong driver! Set to CPU.\n");
            }
        } else if (!strcmp(argv[argc_indx], "-help")) {
            printf("Usage: ./prog_gpu -size L -iters N -driver [CPU|GPU]\n");
            return 0;
        } else {
            argc_indx++;
        }
    }

    size_t NX = size, NY = size, NZ = size;

    double *h_A;
    if ((h_A = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix host_A allocation failed"); exit(1); }

    cpu::init(h_A, size);

    double *d_A, *d_B;
    CHECK_CUDA( cudaMalloc((void**)&d_A, NX * NY * NZ * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**)&d_B, NX * NY * NZ * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(d_A, h_A, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_B, h_A, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )

    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((size-1) / threads_per_block.x + 1,
                                (size-1) / threads_per_block.y + 1,
                                (size-1) / threads_per_block.z + 1);

    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    // TODO: Add some warmup iters
    int it = 0;
    double t1 = 0.0, t2 = 0.0;
    double eps = 0.0;

    if (drv == driver_t::CPU) {
        t1 = timer();

        for (it = 0; it < iters; it++) {
            eps = cpu::adi3d(h_A, size);

            printf(" IT = %4i   EPS = %14.7E\n", it, eps);
            if (eps < max_eps)
                break;
        }

        t2 = timer();
    }

    if (drv == driver_t::GPU) {
        t1 = timer();

        for (it = 0; it < iters; it++) {
            eps = gpu::update_wrapper(d_A, d_B, NX, NY, NZ);

            printf(" IT = %4i   EPS = %14.7E\n", it, eps);
            if (eps < max_eps)
                break;
        }

        CHECK_CUDA( cudaDeviceSynchronize() )
        t2 = timer();
    }

    free(h_A);
    CHECK_CUDA( cudaFree(d_A) )
    CHECK_CUDA( cudaFree(d_B) )

    printf("\n ===================================\n");
    printf(" ADI Benchmark Completed.\n");
    printf(" Final eps       = %1.12E\n", eps);
    printf(" Size            = %4ld x %4ld x %4ld\n", NX, NY, NZ);
    printf(" Iterations      =       %12d\n", it);
    printf(" ADI Time     =       %8.6lf sec\n", t2-t1);
    printf(" Operation type  =     double precision\n");
    printf(" Driver          = %18s\n", driver.c_str());
    printf(" Verification    =       %12s\n", (fabs(eps - EPS) < DIFF ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    printf(" END OF ADI Benchmark\n");
    printf("\n ===================================\n");
    return 0;
}

