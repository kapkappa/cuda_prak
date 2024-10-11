#include "cudadefs.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

namespace gpu {

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
__global__ void get_eps(const double * __restrict__ A, const double * __restrict__ B, size_t NX, size_t NY, size_t NZ, double *eps_out) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    const size_t id = idx + idy * NX + idz * NX * NY;

    const size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;  // thread index in block
    const size_t block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;         // block index in grid

    __shared__ double shared_eps[BLOCKSIZE];    //1-dimensional shared memory

    double tmp = 0.0;

    if (0 < idx && idx < (NX-1) && 0 < idy && idy < (NY-1) && 0 < idz && idz < (NZ-1)) {
        tmp = B[id] - A[id];
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

__global__ void update(const double * __restrict__ A, double *B, size_t NX, size_t NY, size_t NZ) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z; // Z-axis thread id

    const size_t id = idx + idy * NX + idz * NX * NY;

    if (idx == 0 || idx >= (NX-1) || idy == 0 || idy >= (NY-1) || idz == 0 || idz >= (NZ-1)) {
        return;
    }

    size_t offset_x = 1;
    size_t offset_y = NX;
    size_t offset_z = NX * NY;

    B[id] = (A[id - offset_x] + A[id - offset_y] + A[id - offset_z] + A[id + offset_x] + A[id + offset_y] + A[id + offset_z]) / 6.0;

    return;
}


double get_eps_wrapper(double *A, double *B, size_t NX, size_t NY, size_t NZ, double *eps_out) {

    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((NX-1) / threads_per_block.x + 1,
                                (NY-1) / threads_per_block.y + 1,
                                (NZ-1) / threads_per_block.z + 1);

    uint32_t grid_size = blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;

    get_eps<TOTAL_BLOCKSIZE><<<blocks_per_grid, threads_per_block>>>(A, B, NX, NY, NZ, eps_out);

    thrust::device_ptr<double> eps_ptr = thrust::device_pointer_cast(eps_out);

    return sqrt( thrust::reduce(thrust::device, eps_ptr, eps_ptr + grid_size, 0.0) );
}

void update_wrapper(double *A, double *B, size_t NX, size_t NY, size_t NZ) {

    dim3 threads_per_block = dim3(X_BLOCKSIZE, Y_BLOCKSIZE, Z_BLOCKSIZE);
    dim3 blocks_per_grid = dim3((NX-1) / threads_per_block.x + 1,
                                (NY-1) / threads_per_block.y + 1,
                                (NZ-1) / threads_per_block.z + 1);

    update<<<blocks_per_grid, threads_per_block>>>(A, B, NX, NY, NZ);
}


} // gpu
