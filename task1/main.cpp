/* Jacobi-3 program */

#include <string>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include "jac3d.h"
#include "jac3d.cuh"
#include "cudadefs.h"

#define MAX_EPS 5E-1
#define MAX_DIFF 1E-6

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
    size_t size = 30;
    std::string driver = "CPU";
    bool verification = false;
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
            printf("Usage: ./prog_gpu -size L -iters N -driver [CPU|GPU] [-verification]\n");
            return 0;
        } else if (!strcmp(argv[argc_indx], "-verification")) {
            argc_indx++;
            verification = true;
        } else {
            argc_indx++;
        }
    }

    size_t NX = size, NY = size, NZ = size;

    double *h_A, *h_B;
    if ((h_A = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix host_A allocation failed"); exit(1); }
    if ((h_B = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix host_B allocation failed"); exit(2); }

    cpu::init(h_A, h_B, size);


    double *d_A, *d_B;
    CHECK_CUDA( cudaMalloc((void**)&d_A, NX * NY * NZ * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**)&d_B, NX * NY * NZ * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(d_A, h_A, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_B, h_B, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )

    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((size-1) / threads_per_block.x + 1,
                                (size-1) / threads_per_block.y + 1,
                                (size-1) / threads_per_block.z + 1);

    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    double eps = 1.0, *eps_out;
    CHECK_CUDA( cudaMalloc((void**)&eps_out, sizeof(double) * grid_size) )

    double *cpu_eps = NULL, *gpu_eps = NULL;
    if (verification) {
        if ((cpu_eps = (double*)calloc(iters, sizeof(double))) == NULL) { perror("cpu_eps allocation failed"); exit(3); }
        if ((gpu_eps = (double*)calloc(iters, sizeof(double))) == NULL) { perror("gpu_eps allocation failed"); exit(4); }
    }


    // TODO: Add some warmup iters
    int it = 0;
    double t1 = 0.0, t2 = 0.0, t3 = 0.0;

    if (verification || (drv == driver_t::CPU)) {
        t1 = timer();

        for (it = 0; it < iters; it++) {
            std::swap(h_A, h_B);
            cpu::jac3d(h_A, h_B, size);

            if (verification) {
                eps = cpu::get_eps(h_A, h_B, size);
                cpu_eps[it] = eps;
            }
        }

        t2 = timer();

        if (!verification) {
            eps = cpu::get_eps(h_A, h_B, size);
        }

        t3 = timer();
    }

    if (verification || (drv == driver_t::GPU)) {
        t1 = timer();

        for (it = 0; it < iters; it++) {
            std::swap(d_A, d_B);
            gpu::update_wrapper(d_A, d_B, NX, NY, NZ);
            if (verification) {
                gpu_eps[it] = gpu::get_eps_wrapper(d_A, d_B, NX, NY, NZ, eps_out);
            }
        }

        CHECK_CUDA( cudaDeviceSynchronize() )
        t2 = timer();

        if (!verification) {
            eps = gpu::get_eps_wrapper(d_A, d_B, NX, NY, NZ, eps_out);
        }

        CHECK_CUDA( cudaDeviceSynchronize() )
        t3 = timer();
    }


    if (verification) {
        for (int i = 0; i < it; i++) {
            double tmp = fabs(cpu_eps[i] - gpu_eps[i]);
            if (tmp >= MAX_DIFF) {
                printf(" IT = %4i, EPS check failed!\n", i);
                printf("cpu_eps[%i] = %3.11E gpu_eps[%i] = %3.11E, diff = %3.11E\n", i, cpu_eps[i], i, gpu_eps[i], tmp);
            }
        }
        free(cpu_eps);
        free(gpu_eps);
    }

    free(h_A);
    free(h_B);

    CHECK_CUDA( cudaFree(d_A) )
    CHECK_CUDA( cudaFree(d_B) )
    CHECK_CUDA( cudaFree(eps_out) )

    if (verification) {
        printf("\n ===================================\n");
        printf(" Verification Completed.\n");
        printf(" Final eps      = %1.12E\n", eps);
        printf(" Test size      = %4ld x %4ld x %4ld\n", NX, NY, NZ);
        printf(" Test iters     =       %12d\n", iters);
        printf(" Operation type =     floating point\n");
        printf("\n ===================================\n");
    } else {
        printf("\n ===================================\n");
        printf(" Jacobi3D Benchmark Completed.\n");
        printf(" Final eps       = %1.12E\n", eps);
        printf(" Size            = %4ld x %4ld x %4ld\n", NX, NY, NZ);
        printf(" Iterations      =       %12d\n", it);
        printf(" Jacobi Time     =       %8.6lf sec\n", t2-t1);
        printf(" 1 Eps Time      =       %8.6lf sec\n", t3-t2);
        printf(" Operation type  =     floating point\n");
        printf(" Driver          = %18s\n", driver.c_str());
        printf(" END OF Jacobi3D Benchmark\n");
        printf("\n ===================================\n");
    }
    return 0;
}

