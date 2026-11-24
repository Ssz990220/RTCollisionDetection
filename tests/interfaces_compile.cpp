// Compile check of the new stack's interface headers (Stage 0-B, wired into CMake in Stage 0-C).
// Verifies that the headers under include/rtcd/ compile as plain C++20 WITHOUT any CUDA or OptiX include path.
// Built by the OBJECT target rtcd_interfaces_compile (tests/CMakeLists.txt, part of AllNew) with only include/ on
// the include path and /W4 /WX /permissive-; nothing here is linked or run.
// This TU checks that all headers compile together and that the declarations are usable (an implementable
// Backend, constexpr helpers, aggregate initialization). Self-containment of EACH header (compiles when included
// first and alone) is checked by the generated TUs of rtcd_headers_selfcontained (same file); see
// docs/design/interfaces.md, section "Compile check".

#include <rtcd/limits.h>
#include <rtcd/types.h>
#include <rtcd/errors.h>
#include <rtcd/device_buffer.h>
#include <rtcd/context.h>
#include <rtcd/geometry.h>
#include <rtcd/config.h>
#include <rtcd/scene.h>
#include <rtcd/robot_model.h>
#include <rtcd/backend.h>
#include <rtcd/checker.h>
#include <rtcd/json_config.h>
#include <rtcd/rtcd.h>
#include <rtcd/internal/algorithms.h>
#include <rtcd/internal/accel.h>
#include <rtcd/internal/kinematics.h>
#include <rtcd/internal/reference_executor.h>

#include <filesystem>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

    // A minimal backend: the interface must be implementable as declared.
    class NullBackend final : public rtcd::Backend {
    public:
        rtcd::Mode mode() const noexcept override { return rtcd::Mode::Ray; }
        const rtcd::ProgramSet& programs() const noexcept override { return programs_; }
        void beginCheck(const rtcd::CheckInfo&) override {}
        rtcd::LaunchPlan prepare(const rtcd::StreamContext&, const rtcd::BatchView& batch) override {
            params_.hitCounts = batch.hitCounts.ptr;
            rtcd::LaunchPlan plan;
            plan.width       = batch.itemCount;
            plan.params      = &params_;
            plan.paramsBytes = sizeof(params_);
            return plan;
        }
        bool synchronizesHost(std::uint32_t) const noexcept override { return false; }

    private:
        rtcd::ProgramSet programs_;
        rtcd::RobotRayParams params_{};
    };

    static_assert(!std::is_copy_constructible_v<rtcd::DeviceBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<rtcd::DeviceBuffer>);
    static_assert(!std::is_copy_constructible_v<rtcd::CollisionChecker>);
    static_assert(std::is_move_constructible_v<rtcd::CollisionChecker>);
    static_assert(std::is_trivially_copyable_v<rtcd::KinematicsTable>);
    static_assert(std::is_trivially_copyable_v<rtcd::Obb>);
    static_assert(std::is_trivially_copyable_v<rtcd::EdgeRayParams>);
    static_assert(std::is_trivially_copyable_v<rtcd::RobotRayParams>);
    static_assert(std::is_standard_layout_v<rtcd::KinematicsTable>);
    static_assert(alignof(rtcd::SbtRecord) == rtcd::kSbtRecordAlignment);
    static_assert(rtcd::isContinuous(rtcd::Mode::QuadraticCurve) && !rtcd::isContinuous(rtcd::Mode::Mesh));
    static_assert(rtcd::curveDegree(rtcd::Mode::CubicCurve) == 3 && rtcd::isSupportedCurveDegree(3));
    static_assert(rtcd::toBuildFlags(rtcd::BuildPreference::FastTraceCompact)
                  == (rtcd::kBuildFlagPreferFastTrace | rtcd::kBuildFlagAllowCompaction));
    static_assert(rtcd::ItemLayout{7, 16}.floatsPerItem() == 112);
    static_assert(std::is_convertible_v<rtcd::DeviceSpan<float>, rtcd::DeviceSpan<const float>>);

    [[maybe_unused]] void useDeclarations() {
        NullBackend b;
        rtcd::BatchView batch;
        batch.itemCount = 4;
        [[maybe_unused]] rtcd::LaunchPlan plan = b.prepare(rtcd::StreamContext{}, batch);

        rtcd::CheckerConfig cfg;
        cfg.mode                   = rtcd::Mode::QuadraticCurve;
        cfg.curve.refit.kind       = rtcd::RefitPolicy::Kind::Every;
        cfg.curve.refit.every      = 4;
        cfg.sceneEdges             = rtcd::EdgeSet::Loop;
        [[maybe_unused]] const bool autoGeometryFlags = !cfg.curve.geometryFlags.has_value(); // default "auto"
        cfg.curve.geometryFlags    = rtcd::kGeometryFlagRequireSingleAnyHitCall;
        [[maybe_unused]] rtcd::ItemLayout layout = cfg.itemLayout(7);

        // rtcd_geometry seam (declarations only; this TU is never linked).
        [[maybe_unused]] rtcd::TriangleMesh (*loadMesh)(const std::filesystem::path&, const rtcd::Transform&, float) =
            &rtcd::loadObjMesh;
        [[maybe_unused]] void (*edges)(rtcd::TriangleMesh&)                            = &rtcd::computeEdges;
        [[maybe_unused]] rtcd::Obb (*fit)(std::span<const rtcd::Vec3>)                   = &rtcd::fitObb;
        [[maybe_unused]] std::vector<rtcd::Sphere> (*spheres)(const std::filesystem::path&, std::string_view) =
            &rtcd::loadSpheres;

        rtcd::Transform t = rtcd::Transform::fromXyzRpy(0.45, 0, 0.4, 0, 0, 0) * rtcd::Transform::translation(0, 1, 0);
        [[maybe_unused]] rtcd::Vec3 p = t.apply({1, 2, 3});

        rtcd::SceneDesc scene;
        scene.obstacles.push_back(rtcd::ObstacleDesc{"shelf", "shelves.obj", t, 1.0f});

        rtcd::RobotDesc robot;
        robot.description = "panda.urdf";
        robot.spheres     = rtcd::GeometrySource{"spheres", "xyzr", {}};

        rtcd::Logger logger;
        [[maybe_unused]] bool on = logger.enabled(rtcd::LogLevel::Error);

        rtcd::HostItems items{std::span<const float>(), 0, 0};
        (void) items;

        try {
            throw rtcd::CapacityError("too many items", "max_items_per_check", 9000, 8192);
        } catch (const rtcd::Error& e) {
            (void) e.kind();
        }
    }

} // namespace
