#pragma once

#include <cuda_runtime_api.h>

#ifndef X_BLOCKSIZE
#define X_BLOCKSIZE 32
#endif
#ifndef Y_BLOCKSIZE
#define Y_BLOCKSIZE 4
#endif
#ifndef Z_BLOCKSIZE
#define Z_BLOCKSIZE 1
#endif

#define TOTAL_BLOCKSIZE (X_BLOCKSIZE * Y_BLOCKSIZE)


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
