#pragma once

// optix 7
#include "CUDAMacro.h"
#include "optix_stubs.h"
#include <iostream>
#include <optix.h>

// Checked in all build types: the check is a single host-side branch per OptiX API call, and
// silently ignoring failures (e.g. a module that fails to compile) makes errors very hard to trace.
#define OPTIX_CHECK(call)                                                                                    \
    {                                                                                                        \
        OptixResult res = call;                                                                              \
        if (res != OPTIX_SUCCESS) {                                                                          \
            fprintf(stderr, "Optix call (%s) failed with %s (%d) (%s:%d)\n", #call, optixGetErrorName(res), \
                res, __FILE__, __LINE__);                                                                    \
            exit(2);                                                                                         \
        }                                                                                                    \
    }


#define ASSERT_FAIL_MSG(msg) assertFailMsg(msg, __FILE__, __LINE__)


#define OPTIX_STRINGIFY2(name) #name
#define OPTIX_STRINGIFY(name)  OPTIX_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME      OPTIX_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
#define OPTIX_SAMPLE_DIR       OPTIX_STRINGIFY(OPTIX_SAMPLE_DIR_DEFINE)
