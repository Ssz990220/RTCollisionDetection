// ABI check (compile only, target rtcd_core_abi_check in tests/core/CMakeLists.txt): the rtcd handle aliases and
// mirrored enums of the public headers equal the CUDA / OptiX ones. Written in Stage 0-B; owned by Wave 1 a.
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_types.h>
#include <rtcd/rtcd.h>
#include <rtcd/backend.h>
#include <type_traits>

static_assert(std::is_same_v<rtcd::StreamHandle, cudaStream_t>);
static_assert(std::is_same_v<rtcd::StreamHandle, CUstream>);
static_assert(std::is_same_v<rtcd::CudaContextHandle, CUcontext>);
static_assert(std::is_same_v<rtcd::OptixContextHandle, OptixDeviceContext>);
static_assert(std::is_same_v<rtcd::DevicePtr, CUdeviceptr>);
static_assert(std::is_same_v<rtcd::TraversableHandle, OptixTraversableHandle>);
static_assert(rtcd::kBuildFlagAllowUpdate == OPTIX_BUILD_FLAG_ALLOW_UPDATE);
static_assert(rtcd::kBuildFlagAllowCompaction == OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
static_assert(rtcd::kBuildFlagPreferFastTrace == OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
static_assert(rtcd::kBuildFlagPreferFastBuild == OPTIX_BUILD_FLAG_PREFER_FAST_BUILD);
static_assert(rtcd::kGeometryFlagDisableAnyHit == OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
static_assert(rtcd::kGeometryFlagRequireSingleAnyHitCall == OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
static_assert(rtcd::kPrimitiveRoundQuadratic == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE);
static_assert(rtcd::kPrimitiveRoundCubic == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE);
static_assert(rtcd::kPrimitiveRoundLinear == OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR);
static_assert(rtcd::kPrimitiveSphere == OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);
static_assert(rtcd::kPrimitiveTriangle == static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE));
static_assert(static_cast<unsigned>(rtcd::TraversableGraph::SingleGas) == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS);
static_assert(static_cast<unsigned>(rtcd::TraversableGraph::SingleLevelInstancing)
              == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING);
static_assert(rtcd::OptixOptions::kExceptionStackOverflow == OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW);
static_assert(rtcd::OptixOptions::kExceptionTraceDepth == OPTIX_EXCEPTION_FLAG_TRACE_DEPTH);
static_assert(rtcd::OptixOptions::kExceptionUser == OPTIX_EXCEPTION_FLAG_USER);
static_assert(sizeof(OptixInstance) == 80);
