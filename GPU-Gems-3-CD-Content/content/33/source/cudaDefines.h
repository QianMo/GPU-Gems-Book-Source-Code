#ifndef _CUDA_DEFINES_
#define _CUDA_DEFINES_

// this is a shortened version of the cutil lib from the CUDA SDK

#include "cuda.h"

extern "C" void cutCheckBankAccess( unsigned int tidx, unsigned int tidy, unsigned int tidz,
									unsigned int bdimx, unsigned int bdimy, unsigned int bdimz,
									const char* file, const int line, const char* aname,
									const int index);

#ifdef _DEBUG

#  define CU_SAFE_CALL( call ) do {                                          \
    CUresult err = call;                                                     \
    if( CUDA_SUCCESS != err) {                                               \
        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)

#  define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#else  // not DEBUG

    // void macros for performance reasons
#  define CU_SAFE_CALL( call) call
#  define CUDA_SAFE_CALL( call) call
#  define CUT_CHECK_ERROR(errorMessage)

#endif

#if __DEVICE_EMULATION__

#  define CUT_CHECK_DEVICE()

    // Interface for bank conflict checker
#define CUT_BANK_CHECKER( array, index)                                      \
    (cutCheckBankAccess( threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x,    \
    blockDim.y, blockDim.z,                               \
    __FILE__, __LINE__, #array, index ),                  \
    array[index])

#else

#  define CUT_CHECK_DEVICE() do {                                            \
    int deviceCount;                                                         \
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));                        \
    if (deviceCount == 0) {                                                  \
        fprintf(stderr, "There is no device.\n");                            \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    int dev;                                                                 \
    for (dev = 0; dev < deviceCount; ++dev) {                                \
        cudaDeviceProp deviceProp;                                           \
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));           \
        if (strncmp(deviceProp.name, "Device Emulation", 16))                \
            break;                                                           \
    }                                                                        \
    if (dev == deviceCount) {                                                \
        fprintf(stderr, "There is no device supporting CUDA.\n");            \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    else                                                                     \
        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  \
} while (0)

#define CUT_BANK_CHECKER( array, index)  array[index]

#endif


#endif  // #ifndef _CUDA_DEFINES_
