/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define MAX_EPS 5E-1
#define MAX_DIFF 1E-6

#ifndef X_BLOCKSIZE
#define X_BLOCKSIZE 4
#endif
#ifndef Y_BLOCKSIZE
#define Y_BLOCKSIZE 2
#endif
#ifndef Z_BLOCKSIZE
#define Z_BLOCKSIZE 4
#endif

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

// For square matrices only!
static inline size_t get_index(size_t i, size_t j, size_t k, size_t size) {
    return i * size * size + j * size + k;
}


/////////////////////////////////////////////////////////////////////


template <uint32_t BLOCKSIZE>
__device__ __forceinline__ void warp_reduce(size_t i, volatile double* data) {
    if (BLOCKSIZE >= 64) data[i] += data[i + 32];
    if (BLOCKSIZE >= 32) data[i] += data[i + 16];
    if (BLOCKSIZE >= 16) data[i] += data[i +  8];
    if (BLOCKSIZE >=  8) data[i] += data[i +  4];
    if (BLOCKSIZE >=  4) data[i] += data[i +  2];
    if (BLOCKSIZE >=  2) data[i] += data[i +  1];
}

template <uint32_t BLOCKSIZE>
__global__ void get_eps(cudaPitchedPtr d_A, cudaPitchedPtr d_B, size_t NX, size_t NY, size_t NZ, double *eps_out) {

    char * A = (char*)d_A.ptr;
    char * B = (char*)d_B.ptr;

    size_t step_y = d_A.pitch;
	size_t step_z = d_A.pitch * NX;

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    const size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;  // thread index in block
    const size_t block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;         // block index in grid

    __shared__ double shared_eps[BLOCKSIZE];    //1-dimensional shared memory

    double tmp = 0.0;

    if (0 < idx && idx < (NX-1) && 0 < idy && idy < (NY-1) && 0 < idz && idz < (NZ-1)) {
        tmp = ((double*)(B + idy * step_y + idz * step_z))[idx] - ((double*)(A + idy * step_y + idz * step_z))[idx];
    }

    shared_eps[thread_id] = tmp * tmp;

    __syncthreads();

//  Unroll block-wise reduction
    if (BLOCKSIZE >= 512) { if (thread_id < 256) { shared_eps[thread_id] += shared_eps[thread_id + 256]; } __syncthreads(); }
    if (BLOCKSIZE >= 256) { if (thread_id < 128) { shared_eps[thread_id] += shared_eps[thread_id + 128]; } __syncthreads(); }
    if (BLOCKSIZE >= 128) { if (thread_id <  64) { shared_eps[thread_id] += shared_eps[thread_id +  64]; } __syncthreads(); }

    if (thread_id < 32) { warp_reduce<BLOCKSIZE>(thread_id, shared_eps); }

    if (thread_id == 0) {
        eps_out[block_id] = shared_eps[0];
    }

    return;
}

__global__ void update(cudaPitchedPtr d_A, cudaPitchedPtr d_B, size_t NX, size_t NY, size_t NZ) {

    char * A = (char *)d_A.ptr;
    char * B = (char *)d_B.ptr;

    size_t step_y = d_A.pitch;
	size_t step_z = d_A.pitch * NX;

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    char* A_id = A + idy * step_y + idz * step_z;
    char* B_id = B + idy * step_y + idz * step_z;

    if (idx == 0 || idx >= (NX-1) || idy == 0 || idy >= (NY-1) || idz == 0 || idz >= (NZ-1)) {
        return;
    }

    ((double*)B_id)[idx] = ( ((double*)(A_id))[idx + 1] + ((double*)(A_id))[idx - 1] +
                             ((double*)(A_id + step_y))[idx] + ((double*)(A_id - step_y))[idx] +
                             ((double*)(A_id + step_z))[idx] + ((double*)(A_id - step_z))[idx] ) / 6.0;
}


/////////////////////////////////////////////////////////////////////


double get_eps(const double *__restrict__ A, const double *__restrict__ B, size_t size) {
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

    for (size_t i = 1; i < NX-1; i++) {
        for (size_t j = 1; j < NY-1; j++) {
            for (size_t k = 1; k < NZ-1; k++) {
                size_t idx = get_index(i, j, k, size);
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
                B[idx] = (A[idx - offset_k] + A[idx - offset_j] + A[idx - offset_i] +
                          A[idx + offset_k] + A[idx + offset_j] + A[idx + offset_i]) / 6.0;
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////


int main(int argc, char **argv) {

    int argc_indx = 0;
    int iters = 100;
    size_t size = 30;
    std::string driver = "CPU";
    bool verification = false;
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

    cudaExtent extent_bytes = make_cudaExtent(sizeof(double) * NX, NY, NZ);

    cudaPitchedPtr dev_pitched_A, dev_pitched_B;

    CHECK_CUDA( cudaMalloc3D(&dev_pitched_A, extent_bytes) )  // Allocate pitched structure
    CHECK_CUDA( cudaMalloc3D(&dev_pitched_B, extent_bytes) )

    CHECK_CUDA( cudaMemset3D(dev_pitched_A, 0.0, extent_bytes) )  // Init dev_A with 0.0
    CHECK_CUDA( cudaMemset3D(dev_pitched_B, 0.0, extent_bytes) )

    cudaPitchedPtr host_pitched_A = make_cudaPitchedPtr((void*)h_A, sizeof(double) * NX, NY, NZ);
    cudaPitchedPtr host_pitched_B = make_cudaPitchedPtr((void*)h_B, sizeof(double) * NX, NY, NZ);

    cudaMemcpy3DParms params = {0};
    params.extent = extent_bytes;
    params.kind   = cudaMemcpyHostToDevice;

    params.srcPtr = host_pitched_A;
    params.dstPtr = dev_pitched_A;

    cudaMemcpy3D(&params);

    params.srcPtr = host_pitched_B;
    params.dstPtr = dev_pitched_B;

    cudaMemcpy3D(&params);


    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((size-1) / threads_per_block.x + 1,
                                (size-1) / threads_per_block.y + 1,
                                (size-1) / threads_per_block.z + 1);

    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    double eps = 1.0, *eps_out;
    CHECK_CUDA( cudaMalloc(&eps_out, sizeof(double) * grid_size) )

    int it = 0;
    double t1 = 0.0, t2 = 0.0, t3 = 0.0;
    float time1 = 0.0, time2 = 0.0;

    double *cpu_eps = NULL, *gpu_eps = NULL;
    if (verification) {
        if ((cpu_eps = (double*)calloc(iters, sizeof(double))) == NULL) { perror("cpu_eps allocation failed"); exit(3); }
        if ((gpu_eps = (double*)calloc(iters, sizeof(double))) == NULL) { perror("gpu_eps allocation failed"); exit(4); }
    }


    if (verification || (drv == driver_t::CPU)) {
        t1 = timer();

        for (it = 0; it < iters; it++) {
            jac3d(h_A, h_B, size);

            if (verification) {
                eps = get_eps(h_A, h_B, size);
                cpu_eps[it] = eps;
            }
        }

        t2 = timer();
        time1 = t2-t1;

        if (!verification) {
            eps = get_eps(h_A, h_B, size);
        }

        t3 = timer();
        time2 = t3-t2;
    }


    if (verification || (drv == driver_t::GPU)) {

        cudaEvent_t start, stop;

        CHECK_CUDA( cudaEventCreate(&start) )
        CHECK_CUDA( cudaEventCreate(&stop) )

        CHECK_CUDA( cudaEventRecord(start, 0) )

        params.dstPtr = dev_pitched_A;
        params.srcPtr = dev_pitched_B;

        for (it = 0; it < iters; it++) {
            cudaMemcpy3D(&params);
            update<<<blocks_per_grid, threads_per_block>>>(dev_pitched_A, dev_pitched_B, NX, NY, NZ);

            if (verification) {
                get_eps<TOTAL_BLOCKSIZE><<<blocks_per_grid, threads_per_block>>>(dev_pitched_A, dev_pitched_B, NX, NY, NZ, eps_out);
                thrust::device_ptr<double> eps_ptr = thrust::device_pointer_cast(eps_out);
                eps = sqrt( thrust::reduce(thrust::device, eps_ptr, eps_ptr + grid_size, 0.0) );
                gpu_eps[it] = eps;
            }
        }

        CHECK_CUDA( cudaEventRecord(stop, 0) )

        CHECK_CUDA( cudaEventSynchronize(stop) )
        CHECK_CUDA( cudaEventElapsedTime(&time1, start, stop) )

        time1 = time1 / 1000.0;

        CHECK_CUDA( cudaEventRecord(start, 0) )

        if (!verification) {
            get_eps<TOTAL_BLOCKSIZE><<<blocks_per_grid, threads_per_block>>>(dev_pitched_A, dev_pitched_B, NX, NY, NZ, eps_out);
            thrust::device_ptr<double> eps_ptr = thrust::device_pointer_cast((double *)eps_out);
            eps = sqrt( thrust::reduce(thrust::device, eps_ptr, eps_ptr + grid_size, 0.0) );
        }

        CHECK_CUDA( cudaEventRecord(stop, 0) )
        CHECK_CUDA( cudaEventSynchronize(stop) )
        CHECK_CUDA( cudaEventElapsedTime(&time2, start, stop) )

        time2 = time2 / 1000.0;

        CHECK_CUDA( cudaEventDestroy(start) )
        CHECK_CUDA( cudaEventDestroy(stop) )
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

    CHECK_CUDA( cudaFree(dev_pitched_A.ptr) )
    CHECK_CUDA( cudaFree(dev_pitched_B.ptr) )
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
        printf(" Jacobi Time     =       %8.6lf sec\n", time1);
        printf(" 1 Eps Time      =       %8.6lf sec\n", time2);
        printf(" Operation type  =     floating point\n");
        printf(" Driver          = %18s\n", driver.c_str());
        printf(" END OF Jacobi3D Benchmark\n");
        printf("\n ===================================\n");
    }
    return 0;
}
