#pragma once
#include "optix_function_table.h"
#include "optix_function_table_definition.h"
#include <CollisionDetector/OptixCompileConfig.h>
#include <Utils/optix7.h>
#include <array>
#include <tuple>

void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

void defaultInitOptix() {
    CUDA_CHECK(cudaFree(0));
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        std::cout << "No CUDA capable devices found" << std::endl;
        exit(1);
    }
    cudaDeviceProp deviceProps;
    // std::cout << "Found " << numDevices << " CUDA devices" << std::endl;
    for (int i = 0; i < numDevices; i++) {
        cudaGetDeviceProperties(&deviceProps, i);
        // std::cout << "Device " << i << ": " << deviceProps.name << std::endl;
    }
    OPTIX_CHECK(optixInit());
    // std::cout << "Optix initialized" << std::endl;
}

// Context options come from the context fields of RTCD::OptixCompileConfig (validation mode, log callback level).
// The default argument keeps the former behaviour: validation only in Debug builds, log callback only with PRINTIT.
void createDefaultContext(CUcontext& cudaContext, OptixDeviceContext& optixContext,
    const RTCD::OptixCompileConfig& cfg = RTCD::OptixCompileConfig::buildDefault()) {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceID);
    // std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
    }

    OptixDeviceContextOptions options = {};
    if (cfg.contextValidation) {
        std::cout << "OptiX validation mode" << std::endl;
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    if (cfg.contextLogLevel > 0) {
        OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, cfg.contextLogLevel));
    }
}

std::tuple<CUcontext, OptixDeviceContext> createContext(
    const RTCD::OptixCompileConfig& cfg = RTCD::OptixCompileConfig::buildDefault()) {
    defaultInitOptix();
    CUcontext cudaContext;
    OptixDeviceContext optixContext;
    createDefaultContext(cudaContext, optixContext, cfg);
    return std::make_tuple(cudaContext, optixContext);
}

template <size_t NSTREAM>
std::tuple<CUcontext, OptixDeviceContext, std::array<cudaStream_t, NSTREAM>> createContextStream(
    const RTCD::OptixCompileConfig& cfg = RTCD::OptixCompileConfig::buildDefault()) {
    defaultInitOptix();
    CUcontext cudaContext;
    OptixDeviceContext optixContext;
    createDefaultContext(cudaContext, optixContext, cfg);
    std::array<cudaStream_t, NSTREAM> streams;
    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    return std::make_tuple(cudaContext, optixContext, streams);
}
