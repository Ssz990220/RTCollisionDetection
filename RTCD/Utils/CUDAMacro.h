#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#if defined(_DEBUG) || defined(DEBUG)
#define CUDA_CHECK(call)                                                                             \
    {                                                                                                \
        cudaError_t rc = call;                                                                       \
        if (rc != cudaSuccess) {                                                                     \
            std::stringstream txt;                                                                   \
            cudaError_t err = rc; /*cudaGetLastError();*/                                            \
            txt << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")"; \
            throw std::runtime_error(txt.str());                                                     \
        }                                                                                            \
    }
#else
#define CUDA_CHECK(call) \
    { call; }
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define CUDA_CHECK_LAST(message)                                                                         \
    {                                                                                                    \
        cudaError_t error = cudaGetLastError();                                                          \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(message);                                                           \
        }                                                                                                \
    }
#else
#define CUDA_CHECK_LAST(message) \
    {}
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define CUDA_SYNC_CHECK()                                                                                \
    {                                                                                                    \
        cudaDeviceSynchronize();                                                                         \
        cudaError_t error = cudaGetLastError();                                                          \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(2);                                                                                     \
        }                                                                                                \
    }
#else
#define CUDA_SYNC_CHECK() \
    { cudaDeviceSynchronize(); }
#endif
