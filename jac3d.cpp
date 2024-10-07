/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

static inline double timer() {
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#define Max(a, b) ((a) > (b) ? (a) : (b))

double MAXEPS = 0.5f;

int main(int argc, char **argv) {
    int argc_indx = 0;
    int iters = 100;
    size_t size = 30;
    while (argc_indx < argc) {
        if (!strcmp(argv[argc_indx], "-size")) {
            argc_indx++;
            size = atoi(argv[argc_indx]);
        } else if (!strcmp(argv[argc_indx], "-iters")) {
            argc_indx++;
            iters = atoi(argv[argc_indx]);
        } else if (!strcmp(argv[argc_indx], "-help")) {
            printf("Usage: ./prog_cpu -size L -iters N\n");
            return 0;
        } else {
            argc_indx++;
        }
    }

    size_t NX = size, NY = size, NZ = size;

    double *A, *B;
    if ((A = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix A allocation failed"); exit(1); }
    if ((B = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix B allocation failed"); exit(1); }

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t k = 0; k < NZ; k++) {
                A[i * NY * NZ + j * NZ + k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == NX-1 || j == NY-1 || k == NZ-1) {
                    B[i * NY * NZ + j * NZ + k] = 0;
                } else {
                    B[i * NY * NZ + j * NZ + k] = 4 + i + j + k;
                }
            }
        }
    }

    double eps = 0.0;

    double startt = timer();
    /* iteration loop */
    for (int it = 1; it <= iters; it++) {
        eps = 0;

        for (size_t i = 1; i < NX-1; i++) {
            for (size_t j = 1; j < NY-1; j++) {
                for (size_t k = 1; k < NZ-1; k++) {
                    double tmp = fabs(B[i * NY * NZ + j * NZ + k] - A[i * NY * NZ + j * NZ + k]);
                    eps = Max(tmp, eps);
                    A[i * NY * NZ + j * NZ + k] = B[i * NY * NZ + j * NZ + k];
                }
            }
        }

        for (size_t i = 1; i < NX-1; i++) {
            for (size_t j = 1; j < NY-1; j++) {
                for (size_t k = 1; k < NZ-1; k++) {
                    B[i * NY * NZ + j * NZ + k] = (A[(i-1) * NY * NZ + j * NZ + k] + A[(i+1) * NY * NZ + j * NZ + k] +
                                                   A[i * NY * NZ + (j-1) * NZ + k] + A[i * NY * NZ + (j+1) * NZ + k] +
                                                   A[i * NY * NZ + j * NZ + (k-1)] + A[i * NY * NZ + j * NZ + (k+1)]) / 6.0f;
                }
            }
        }

        printf(" IT = %4i   EPS = %14.12E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    double endt = timer();

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4ld x %4ld x %4ld\n", NX, NY, NZ);
    printf(" Iterations      =       %12d\n", iters);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =     floating point\n");
//    printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    printf(" END OF Jacobi3D Benchmark\n");
    return 0;
}
