#include "profile_config.h"

#include <support/options.h>

#include <rtcd/errors.h>

#include <string>

namespace rtcd::sbs {

    CheckerConfig newConfigForProfile(const OldProfile& p, const std::filesystem::path& splineWeightsDir) {
        CheckerConfig c;
        switch (p.mode) {
        case OldMode::Ias: c.mode = Mode::Mesh; break;
        case OldMode::Ray: c.mode = Mode::Ray; break;
        case OldMode::Linear: c.mode = Mode::LinearCurve; break;
        case OldMode::Quad: c.mode = Mode::QuadraticCurve; break;
        }
        c.numStreams       = static_cast<std::uint32_t>(p.streams);
        c.hostDispatch     = HostDispatch::ThreadPerStream;
        c.maxBatchSize     = static_cast<std::uint32_t>(p.batchSize);
        c.maxItemsPerCheck = static_cast<std::uint32_t>(p.maxItems);
        c.obbCulling       = p.obb ? Toggle::On : Toggle::Off;
        c.optix            = OptixOptions::buildDefault();
        auto build = [&](OldBuild b) {
            switch (b) {
            case OldBuild::Compact: return BuildPreference::FastTraceCompact;
            case OldBuild::FastTrace: return BuildPreference::FastTrace;
            case OldBuild::FastBuild: return BuildPreference::FastBuild;
            case OldBuild::None: break;
            }
            throw InvalidArgument("profile " + p.name + ": build type NONE has no new-stack equivalent for this mode");
        };
        if (c.mode == Mode::Mesh) {
            c.mesh.linkGasBuild = build(p.build);
        } else if (isContinuous(c.mode)) {
            c.curve.gasBuild         = build(p.build);
            c.curve.trajectoryPoints = static_cast<std::uint32_t>(p.trajPts);
            c.curve.controlPoints    = static_cast<std::uint32_t>(p.ctrlPts);
            c.curve.splineWeightsDir = splineWeightsDir;
        }
        return c;
    }

    void applyOldSettings(const OldSettings& s, CheckerConfig& config, ContextOptions& context) {
        if (!s.optixConfig.empty()) {
            support::applyLegacyOptixConfig(s.optixConfig, &config, &context);
        }
        if (s.refitEvery) {
            config.curve.refit = support::parseLegacyRefitEvery(std::to_string(*s.refitEvery));
        }
        if (s.curveGeomFlag) {
            config.curve.geometryFlags = support::parseLegacyCurveGeometryFlag(*s.curveGeomFlag);
        }
    }

} // namespace rtcd::sbs
