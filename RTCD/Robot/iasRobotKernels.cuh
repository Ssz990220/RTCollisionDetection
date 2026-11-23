#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Kernels for the ObsToRobot (IAS robot) hot path.

// upInsVisibility
// @brief For each of the first n instances: copy the link transform (12 floats, Adj layout, one per instance) into
// the instance and set its visibility mask from the OBB broad-phase mask (mask[i] > 0 -> 255, else 0).
// Rays are traced with mask 255, so an instance with visibilityMask 0 is culled by OptiX and never reaches any-hit.
// This replaces stream compaction: the instance array keeps its order (identity instance -> pose map) and the
// instance count is known on the host, so no device-to-host round trip is needed before optixAccelBuild.
// If mask is 0 (null), all instances are made visible.
void upInsVisibility(CUdeviceptr instances, CUdeviceptr transforms, CUdeviceptr mask, size_t n, cudaStream_t stream);
