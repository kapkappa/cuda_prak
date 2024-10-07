/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define MAXEPS 0.5

#define X_BLOCKSIZE 8
#define Y_BLOCKSIZE 8
#define Z_BLOCKSIZE 8
#define TOTAL_BLOCKSIZE (X_BLOCKSIZE * Y_BLOCKSIZE * Z_BLOCKSIZE)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

static inline double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

static inline size_t get_index(size_t i, size_t j, size_t k, size_t size) {
    return i * size * size + j * size + k;
}

template <uint32_t BLOCKSIZE>
__device__ __forceinline__ void block_reduce_max(size_t i, double* data) {
    if (BLOCKSIZE > 512) { if (i < 512 && i + 512 < BLOCKSIZE) { data[i] = Max(data[i], data[i + 512]); } __syncthreads(); }
    if (BLOCKSIZE > 256) { if (i < 256 && i + 256 < BLOCKSIZE) { data[i] = Max(data[i], data[i + 256]); } __syncthreads(); }
    if (BLOCKSIZE > 128) { if (i < 128 && i + 128 < BLOCKSIZE) { data[i] = Max(data[i], data[i + 128]); } __syncthreads(); }
    if (BLOCKSIZE >  64) { if (i <  64 && i +  64 < BLOCKSIZE) { data[i] = Max(data[i], data[i +  64]); } __syncthreads(); }
    if (BLOCKSIZE >  32) { if (i <  32 && i +  32 < BLOCKSIZE) { data[i] = Max(data[i], data[i +  32]); } __syncthreads(); }
    if (BLOCKSIZE >  16) { if (i <  16 && i +  16 < BLOCKSIZE) { data[i] = Max(data[i], data[i +  16]); } __syncthreads(); }
    if (BLOCKSIZE >   8) { if (i <   8 && i +   8 < BLOCKSIZE) { data[i] = Max(data[i], data[i +   8]); } __syncthreads(); }
    if (BLOCKSIZE >   4) { if (i <   4 && i +   4 < BLOCKSIZE) { data[i] = Max(data[i], data[i +   4]); } __syncthreads(); }
    if (BLOCKSIZE >   2) { if (i <   2 && i +   2 < BLOCKSIZE) { data[i] = Max(data[i], data[i +   2]); } __syncthreads(); }
    if (BLOCKSIZE >   1) { if (i <   1 && i +   1 < BLOCKSIZE) { data[i] = Max(data[i], data[i +   1]); } __syncthreads(); }
}

template <uint32_t BLOCKSIZE>
__global__ void jacobi(double *A, double *B, size_t NX, size_t NY, size_t NZ, double *eps_out) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    const size_t id = idx + idy * NX + idz * NX * NY;

    const size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;  // thread index in block
    const size_t block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;         // block index in grid

    __shared__ double shared_eps[BLOCKSIZE];    //1-dimensional shared memory

    double tmp = 0.0;

    if (0 < idx && idx < (NX-1) && 0 < idy && idy < (NY-1) && 0 < idz && idz < (NZ-1)) {
        tmp = fabs(B[id] - A[id]);
        A[id] = B[id];
    }

    shared_eps[thread_id] = tmp;

    __syncthreads();

    block_reduce_max<BLOCKSIZE>(thread_id, shared_eps);

    if (thread_id == 0) {
        eps_out[block_id] = shared_eps[0];
    }

    if (idx == 0 || idx >= (NX-1) || idy == 0 || idy >= (NY-1) || idz == 0 || idz >= (NZ-1)) {
        return;
    }

    size_t offset_x = 1;
    size_t offset_y = NX;
    size_t offset_z = NX * NY;

    B[id] = (A[id - offset_x] + A[id - offset_y] + A[id - offset_z] + A[id + offset_x] + A[id + offset_y] + A[id + offset_z]) / 6.0;
}

double solve(double *A, double *B, size_t size) {
    size_t NX = size, NY = size, NZ = size;
    double eps = 0.0;

    for (size_t i = 1; i < NX-1; i++) {
        for (size_t j = 1; j < NY-1; j++) {
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                double tmp = fabs(B[idx] - A[idx]);
                eps = Max(tmp, eps);
                A[idx] = B[idx];
            }
        }
    }

    size_t offset_i = NZ * NY;
    size_t offset_j = NZ;
    size_t offset_k = 1;

    for (size_t i = 1; i < NX-1; i++) {
        for (size_t j = 1; j < NY-1; j++) {
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
                B[idx] = (A[idx - offset_i] + A[idx - offset_j] + A[idx - offset_k] +
                          A[idx + offset_i] + A[idx + offset_j] + A[idx + offset_k]) / 6.0;
            }
        }
    }

    return eps;
}

int main(int argc, char **argv) {

    int argc_indx = 0;
    int iters = 100;
    size_t size = 30;
    std::string driver = "CPU";
    enum class driver_t {CPU, GPU} drv = driver_t::CPU;
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
            printf("Usage: ./prog_gpu -size L -iters N\n");
            return 0;
        } else {
            argc_indx++;
        }
    }

    size_t NX = size, NY = size, NZ = size;

    double *h_A, *h_B;

    if ((h_A = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix host_A allocation failed"); exit(1); }
    if ((h_B = (double*)malloc(sizeof(double) * NX * NY * NZ)) == NULL) { perror("matrix host_B allocation failed"); exit(1); }

    // Init
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t k = 0; k < NZ; k++) {
                size_t idx = get_index(i, j, k, size);
                h_A[idx] = 0;
                if (i == 0 || j == 0 || k == 0 || i == NX-1 || j == NY-1 || k == NZ-1) {
                    h_B[idx] = 0.0;
                } else {
                    h_B[idx] = 4.0 + i + j + k;
                }
            }
        }
    }

    double *d_A, *d_B;
    CHECK_CUDA( cudaMalloc(&d_A, NX * NY * NZ * sizeof(double)) )
    CHECK_CUDA( cudaMalloc(&d_B, NX * NY * NZ * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(d_A, h_A, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_B, h_B, sizeof(double) * NX * NY * NZ, cudaMemcpyHostToDevice) )

    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((size-1) / threads_per_block.x + 1,
                                (size-1) / threads_per_block.y + 1,
                                (size-1) / threads_per_block.z + 1);

//    constexpr int block_size = threads_per_block.x * threads_per_block.y * threads_per_block.z;
    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    double eps = 0.0, *eps_out;
    cudaMalloc(&eps_out, sizeof(double) * grid_size);

    double t1 = timer();

    for (int it = 1; it <= iters; it++) {

        if (drv == driver_t::GPU) {

            jacobi<TOTAL_BLOCKSIZE><<<blocks_per_grid, threads_per_block>>>(d_A, d_B, NX, NY, NZ, eps_out);

            thrust::device_ptr<double> eps_ptr = thrust::device_pointer_cast(eps_out);
            eps = *(thrust::max_element(eps_ptr, eps_ptr + grid_size));

        } else {
            eps = solve(h_A, h_B, size);
        }

        printf(" IT = %4i   EPS = %14.12E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    double t2 = timer();

    free(h_A);
    free(h_B);

    CHECK_CUDA( cudaFree(d_A) )
    CHECK_CUDA( cudaFree(d_B) )
    CHECK_CUDA( cudaFree(eps_out) )

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4ld x %4ld x %4ld\n", NX, NY, NZ);
    printf(" Iterations      =       %12d\n", iters);
    printf(" Time in seconds =       %12.6lf\n", t2 - t1);
    printf(" Operation type  =     floating point\n");
    printf(" Driver          = %s\n", driver.c_str());
//    printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));
    printf(" END OF Jacobi3D Benchmark\n");
    return 0;
}
