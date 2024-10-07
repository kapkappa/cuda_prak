/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


#define Max(a, b) ((a) > (b) ? (a) : (b))

#define L 384
#define ITMAX 100
#define MAXEPS 0.5f

static inline double timer() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
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
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    size_t id = idx + idy * NX + idz * NX * NY;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    size_t block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    __shared__ double shared_eps[BLOCKSIZE];    //1-dimensional shared memory

    double eps = 0;
    double tmp = fabs(B[id] - A[id]);
    shared_eps[thread_id] = Max(tmp, eps);
    A[id] = B[id];

    block_reduce_max<BLOCKSIZE>(thread_id, shared_eps);

    if (thread_id == 0) {
        eps_out[block_id] = shared_eps[0];
    }

    if (idx == 0 || idx >= NX-1 || idy == 0 || idy >= NY-1 || idz == 0 || idz >= NZ-1) {
        return;
    }

    size_t offset_x = 1;
    size_t offset_y = NX;
    size_t offset_z = NX * NY;

    B[id] = (A[id - offset_x] + A[id - offset_y] + A[id - offset_z] + A[id + offset_x] + A[id + offset_y] + A[id + offset_z]) / 6.0;
}



int main(int argc, char **argv) {

//    size_t size = 0;
//    if (argc == 2) {
//        size = atoi(argv[1]);
//    }

    size_t size = argc == 2 ? atoi(argv[1]) : 0;

    double *h_A, *h_B;

    h_A = (double*)malloc(sizeof(double) * size * size * size);
    h_B = (double*)malloc(sizeof(double) * size * size * size);


//    double A[L][L][L], B[L][L][L];
//    double startt, endt;

    // Init
    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < L; j++) {
            for (size_t k = 0; k < L; k++) {
                h_A[i * size * size + j * size + k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) {
                    h_B[i * size * size + j * size + k] = 0;
                } else {
                    h_B[i * size * size + j * size + k] = 4 + i + j + k;
                }
            }
        }
    }

    double *d_A, *d_B;
    cudaMalloc(&d_A, size * size * size);
    cudaMalloc(&d_B, size * size * size);

    cudaMemcpy(d_A, h_A, size*size*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size*size*size, cudaMemcpyHostToDevice);

    dim3 threads_per_block = dim3(8, 8, 8);
    dim3 blocks_per_grid = dim3((size-1) / threads_per_block.x + 1,
                                (size-1) / threads_per_block.y + 1,
                                (size-1) / threads_per_block.z + 1);

//    constexpr int block_size = threads_per_block.x * threads_per_block.y * threads_per_block.z;
    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    double eps, *eps_out;
    cudaMalloc(&eps_out, sizeof(double) * grid_size);

    double t1 = timer();

    for (int it = 1; it <= ITMAX; it++) {
        jacobi<512><<<blocks_per_grid, threads_per_block>>>(d_A, d_B, size, size, size, eps_out);

        thrust::device_ptr<double> eps_ptr = thrust::device_pointer_cast(eps_out);
        eps = *(thrust::max_element(eps_ptr, eps_ptr + grid_size));

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    double t2 = timer();

/*
    for (it = 1; it <= ITMAX; it++)
    {
        eps = 0;

        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                {
                    double tmp = fabs(B[i][j][k] - A[i][j][k]);
                    eps = Max(tmp, eps);
                    A[i][j][k] = B[i][j][k];
                }

        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                    B[i][j][k] = (A[i - 1][j][k] + A[i][j - 1][k] + A[i][j][k - 1] + A[i][j][k + 1] + A[i][j + 1][k] + A[i + 1][j][k]) / 6.0f;

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }
*/

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    //TODO
    printf(" Time in seconds =       %12.2lf\n", t1 - t2);
    printf(" Operation type  =     floating point\n");
    printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    printf(" END OF Jacobi3D Benchmark\n");
    return 0;
}
