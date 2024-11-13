#include "cudadefs.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace gpu {

#define Max(a, b) ((a) > (b) ? (a) : (b))

template <uint32_t BLOCKSIZE>
__device__ __forceinline__ void warp_reduce_max(size_t i, volatile double* data) {
    if (BLOCKSIZE >= 64) { data[i] = Max(data[i], data[i + 32]); __syncthreads(); }
    if (BLOCKSIZE >= 32) { data[i] = Max(data[i], data[i + 16]); __syncthreads(); }
    if (BLOCKSIZE >= 16) { data[i] = Max(data[i], data[i +  8]); __syncthreads(); }
    if (BLOCKSIZE >=  8) { data[i] = Max(data[i], data[i +  4]); __syncthreads(); }
    if (BLOCKSIZE >=  4) { data[i] = Max(data[i], data[i +  2]); __syncthreads(); }
    if (BLOCKSIZE >=  2) { data[i] = Max(data[i], data[i +  1]); __syncthreads(); }
}

__global__ void update1(double * A, size_t NX, size_t NY, size_t NZ, size_t idx) {

    const size_t idz = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id

    if (idy == 0 || idy >= NX-1 || idz == 0 || idz >= NY-1) {
        return;
    }

    const size_t id = idx + idy * NX + idz * NX * NY;
    A[id] = (A[id + 1] + A[id - 1]) / 2.0;

    return;
}

__global__ void update2(double * A, size_t NX, size_t NY, size_t NZ, size_t idy) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idz = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id

    if (idx == 0 || idx >= (NX-1) || idz == 0 || idz >= NZ-1) {
        return;
    }

    const size_t id = idx + idy * NX + idz * NX * NY;
    A[id] = (A[id + NX] + A[id - NX]) / 2.0;

    return;
}

template <uint32_t BLOCKSIZE>
__global__ void get_eps(double * A, size_t NX, size_t NY, size_t NZ, double *eps_out, size_t idz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // X-axis thread id
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis thread id

    const size_t id = idx + idy * NX + idz * NX * NY;

    const size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;  // thread index in block
    const size_t block_id = blockIdx.x + blockIdx.y * gridDim.x;         // block index in grid

    __shared__ double shared_eps[BLOCKSIZE];    //1-dimensional shared memory

    double tmp = 0.0;

    if (0 < idx && idx < (NX-1) && 0 < idy && idy < (NY-1)) {
        double tmp2 = (A[id + NX*NY] + A[id - NX*NY]) / 2.0;
        tmp = fabs(A[id] - tmp2);
        A[id] = tmp2;
    }

    shared_eps[thread_id] = tmp;

    __syncthreads();

//  Unroll block-wise reduction
    if (BLOCKSIZE >= 512) { if (thread_id < 256) { shared_eps[thread_id] = Max(shared_eps[thread_id], shared_eps[thread_id + 256]); } __syncthreads(); }
    if (BLOCKSIZE >= 256) { if (thread_id < 128) { shared_eps[thread_id] = Max(shared_eps[thread_id], shared_eps[thread_id + 128]); } __syncthreads(); }
    if (BLOCKSIZE >= 128) { if (thread_id <  64) { shared_eps[thread_id] = Max(shared_eps[thread_id], shared_eps[thread_id +  64]); } __syncthreads(); }

    if (thread_id < 32) { warp_reduce_max<BLOCKSIZE>(thread_id, shared_eps); }

    if (thread_id == 0) {
        eps_out[block_id] = shared_eps[0];
    }

    return;
}


double update_wrapper(double *A, size_t NX, size_t NY, size_t NZ) {

    dim3 threads_per_block_2 = dim3(16, 16);
    dim3 blocks_per_grid_2 = dim3((NY-1) / 16 + 1, (NZ-1) / 16 + 1);

    for (int i = 1; i < NX-1; i++) {
        update1<<<blocks_per_grid_2, threads_per_block_2>>>(A, NX, NY, NZ, i);
    }

    for (int j = 1; j < NY-1; j++) {
        update2<<<blocks_per_grid_2, threads_per_block_2>>>(A, NX, NY, NZ, j);
    }

    uint32_t grid_size = blocks_per_grid_2.x * blocks_per_grid_2.y;
    thrust::device_vector<double> eps_out(grid_size);
    thrust::device_ptr<double> eps_ptr = eps_out.data();

    double eps = 0.0;

    for (int k = 1; k < NZ-1; k++) {
        thrust::fill(eps_ptr, eps_ptr + grid_size, 0.0);

        get_eps<16*16><<<blocks_per_grid_2, threads_per_block_2>>>(A, NX, NY, NZ, thrust::raw_pointer_cast(eps_out.data()), k);

        double local_eps = *(thrust::max_element(eps_ptr, eps_ptr + grid_size));

        eps = Max(eps, local_eps);
    }

    return eps;
}


} // gpu
