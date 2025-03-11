#pragma once

// optix 7
#include "CUDAMacro.h"
#include "optix_stubs.h"
#include <iostream>
#include <optix.h>

#if defined(_DEBUG) || defined(DEBUG)
#define OPTIX_CHECK(call)                                                                             \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if (res != OPTIX_SUCCESS) {                                                                   \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                                  \
        }                                                                                             \
    }
#else
#define OPTIX_CHECK(call) \
    { call; }
#endif


#define ASSERT_FAIL_MSG(msg) assertFailMsg(msg, __FILE__, __LINE__)


#define OPTIX_STRINGIFY2(name) #name
#define OPTIX_STRINGIFY(name)  OPTIX_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME      OPTIX_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
#define OPTIX_SAMPLE_DIR       OPTIX_STRINGIFY(OPTIX_SAMPLE_DIR_DEFINE)
