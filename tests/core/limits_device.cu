// Device check (compile only, target rtcd_core_abi_check): rtcd/limits.h must be usable from device code and
// agree with the OptiX constants. Written in Stage 0-B; owned by Wave 1 a.
#include <rtcd/limits.h>
#include <optix_types.h>

static_assert(rtcd::kSbtRecordHeaderSize == OPTIX_SBT_RECORD_HEADER_SIZE, "header size");
static_assert(rtcd::kSbtRecordAlignment == OPTIX_SBT_RECORD_ALIGNMENT, "alignment");
static_assert(sizeof(rtcd::SbtRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0, "record stride");
static_assert(sizeof(OptixTraversableHandle) == sizeof(unsigned long long), "handle");

__global__ void readTable(const rtcd::KinematicsTable* t, const rtcd::Obb* o, float* out) {
    const unsigned j = threadIdx.x % rtcd::kMaxDof;
    out[threadIdx.x] = t->cosMask[j][threadIdx.x % rtcd::kTransformFloats] + t->base[0] + o->axes[2][1]
                     + static_cast<float>(rtcd::kMaxLinks);
}

extern "C" __global__ void readParams(rtcd::EdgeRayParams p, rtcd::RobotRayParams r) {
    reinterpret_cast<unsigned*>(p.hitCounts)[0] = p.edgeCount + r.raysPerRobot;
}
