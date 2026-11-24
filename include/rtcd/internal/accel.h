#pragma once
// rtcd/internal/accel.h -- static acceleration-structure helpers shared by the Scene (obstacle GAS + IAS) and
// the mesh backend (link GAS). Library rtcd_core (Wave 1 a). INTERNAL contract.
//
// These are construction-time helpers: they are SYNCHRONOUS (return when the AS is built and, if requested,
// compacted) and allocate. Per-batch AS builds (robot IAS, curve GAS) stay inside their backends.

#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/types.h>

#include <cstdint>

namespace rtcd::internal {

    struct Accel {
        DeviceBuffer buffer;              // owns the AS memory
        TraversableHandle handle = 0;
    };

    struct TriangleGasInput {
        DeviceSpan<const float> vertices;           // 3 floats per vertex (float3), stride 12 bytes
        DeviceSpan<const std::uint32_t> indices;    // 3 per triangle (uint3), stride 12 bytes
        GeometryFlags geometryFlags = kGeometryFlagNone; // one SBT record, no per-primitive SBT offsets
    };

    // Builds one triangle GAS with `flags`. If flags contains kBuildFlagAllowCompaction the GAS is compacted
    // (emit compacted size, optixAccelCompact when smaller), exactly like the old meshModel::buildGAS.
    // Scene obstacles: flags PREFER_FAST_TRACE | ALLOW_COMPACTION, geometry DISABLE_ANYHIT.
    // Mesh-backend links: flags toBuildFlags(MeshOptions::linkGasBuild), geometry REQUIRE_SINGLE_ANYHIT_CALL.
    Accel buildTriangleGas(const Context& context, const TriangleGasInput& input, BuildFlags flags, StreamHandle stream);

    // Builds an instance AS over `instances` (device array of `count` OptixInstance, 80 bytes each, filled by
    // the caller with OptiX types in its own translation unit).
    Accel buildInstanceAs(const Context& context, DevicePtr instances, std::uint32_t count, BuildFlags flags,
        StreamHandle stream);

} // namespace rtcd::internal
