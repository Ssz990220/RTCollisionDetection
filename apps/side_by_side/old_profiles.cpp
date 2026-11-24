// OLD-STACK TU of the side-by-side runner (deleted with the old stack). The only translation unit of a binary
// that includes old-stack headers (they are not ODR-safe across TUs); compiled once per scene. Every profile is a verbatim copy of an old setup:
//   reg_*        RTCD/CollisionDetector/Test/ccRegression.cpp (NSTREAM 4, REG_BATCHSIZE 128)
//   bm_ias/ray   RTCD/Benchmark/Discrete (NSTREAM 4, batch 1024)
//   bm_linear_*, bm_quad_*, bm_curve_mesh/ray   RTCD/Benchmark/Curve (NSTREAM 16; curves batch 16)
// USE_MULTI_THREADING comes from rtcd_use_old_stack (CMake), as in those executables.
#include "old_runner.h"

#include <CollisionDetector/CCRobotWrapper.h>
#include <CollisionDetector/CollisionDetector.h>
#include <CollisionScenes/batchScene.h>
#include <Robot/batchCurveRobot.h>
#include <Robot/batchIASRobot.h>
#include <Robot/batchRayRobot.h>
#include <Robot/models/pandas.h>

#include <stdexcept>

using namespace RTCD;

// The scene of this library (SHELF / SHELFSIMPLE / DENSESHELF): rtcdSbsBuildOldScene, oldSceneName.
#include "old_scene.inl"

namespace rtcd::sbs {

    namespace {
        template <size_t NS, LinkType TYPE>
        using Wrapper = std::conditional_t<TYPE == LinkType::RAY_ONLINE, CCRayRobot<NS, TYPE>, CCWrappedRobot<NS, TYPE>>;

        template <batchRobotConfig CFG>
        auto makeRobotImpl(std::shared_ptr<robot<Dim>> r, const std::array<cudaStream_t, CFG.NSTREAM>& streams,
            OptixDeviceContext ctx) {
            if constexpr (CFG.TYPE == LinkType::MESH) {
                return std::make_shared<batchIASRobot<CFG>>(r, streams, ctx);
            } else if constexpr (CFG.TYPE == LinkType::RAY_ONLINE) {
                return std::make_shared<batchRayRobot<CFG>>(r, streams, ctx);
            } else {
                return std::make_shared<batchOptixCurveRobot<CFG>>(r, streams, ctx);
            }
        }

        template <batchRobotConfig CFG>
        class Runner final : public OldRunner {
            static constexpr size_t NS      = CFG.NSTREAM;
            static constexpr LinkType TYPE  = CFG.TYPE;
            static constexpr bool CURVE     = TYPE == LinkType::LINEAR_CURVE || TYPE == LinkType::QUADRATIC_CURVE;
            static constexpr batchSceneConfig sceneCfg{false, NS, 0};

        public:
            Runner(const OldProfile& p, OptixDeviceContext ctx, const OldSettings& settings) : profile_(p) {
                OptixCompileConfig ocfg = OptixCompileConfig::buildDefault();
                if (!settings.optixConfig.empty() && !ocfg.setFields(settings.optixConfig)) {
                    throw std::runtime_error("invalid old OptiX config '" + settings.optixConfig + "'");
                }
                for (auto& s : streams_) {
                    CUDA_CHECK(cudaStreamCreate(&s));
                }
                robot_     = std::make_shared<robot<Dim>>(Config);
                robotImpl_ = makeRobotImpl<CFG>(robot_, streams_, ctx);
                if constexpr (CURVE) {
                    if (settings.refitEvery) {
                        robotImpl_->setRefitPolicy(*settings.refitEvery);
                    }
                    if (settings.curveGeomFlag) {
                        if (*settings.curveGeomFlag != "none" && *settings.curveGeomFlag != "single_anyhit") {
                            throw std::runtime_error("invalid curve geometry flag '" + *settings.curveGeomFlag + "'");
                        }
                        robotImpl_->setGeometryFlags(*settings.curveGeomFlag == "none"
                                                         ? OPTIX_GEOMETRY_FLAG_NONE
                                                         : OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
                    }
                } else if (settings.refitEvery || settings.curveGeomFlag) {
                    throw std::runtime_error("refit / curve geometry flag settings only apply to curve profiles");
                }
                ccRobot_ = std::make_shared<Wrapper<NS, TYPE>>(robotImpl_);
                ccRobot_->allocPoseBuffer(p.maxItems);
                scene_      = rtcdSbsBuildOldScene(CURVE); // loop edges for curves (old buildSharedScene(CURVE))
                batchScene_ = std::make_shared<batchScene<sceneCfg>>(scene_, streams_, ctx);
                cc_         = std::make_unique<CollisionDetector<NS, TYPE>>(ctx, streams_);
                cc_->setCompileConfig(ocfg);
                cc_->setupScene(batchScene_);
                cc_->setupOptix();
                cc_->setupRobot(ccRobot_);
                cc_->buildSBT();
                CUDA_SYNC_CHECK();
            }
            ~Runner() override {
                cc_.reset();
                CUDA_SYNC_CHECK();
            }

            const OldProfile& profile() const override { return profile_; }
            void upload(const float* items, std::size_t n) override {
                if (n == 0 || n > profile_.maxItems) {
                    throw std::runtime_error("old runner " + profile_.name + ": item count " + std::to_string(n)
                                             + " outside [1, " + std::to_string(profile_.maxItems) + "]");
                }
                ccRobot_->prepareUpdateData(items, n);
                if constexpr (!CURVE) {
                    cc_->resetGraph();
                }
                cc_->setNCheck(static_cast<uint>(n));
                n_ = n;
                CUDA_SYNC_CHECK();
            }
            void check() override { cc_->check(); }
            std::vector<std::uint32_t> hitCounts(std::size_t n) override {
                if (n > n_) {
                    throw std::runtime_error("old runner: hitCounts(n) beyond the last check");
                }
                std::vector<uint> raw(n, 0);
                cc_->downloadResult(raw);
                return std::vector<std::uint32_t>(raw.begin(), raw.end());
            }

        private:
            OldProfile profile_;
            std::array<cudaStream_t, NS> streams_{};
            std::shared_ptr<robot<Dim>> robot_;
            decltype(makeRobotImpl<CFG>(nullptr, {}, nullptr)) robotImpl_;
            std::shared_ptr<Wrapper<NS, TYPE>> ccRobot_;
            std::shared_ptr<scene> scene_;
            std::shared_ptr<baseBatchScene<NS>> batchScene_;
            std::unique_ptr<CollisionDetector<NS, TYPE>> cc_;
            std::size_t n_ = 0;
        };

        // ---- the compiled-in profiles (configs exactly as in the old executables) ----
        constexpr batchRobotConfig REG_IAS{4, 128, Dim, BuildType::COMPACT, LinkType::MESH, true};
        constexpr batchRobotConfig REG_RAY{4, 128, Dim, BuildType::NONE, LinkType::RAY_ONLINE, true};
        constexpr batchRobotConfig REG_LINEAR{4, 128, Dim, BuildType::FAST_BUILD, LinkType::LINEAR_CURVE, 16, 16, false};
        constexpr batchRobotConfig REG_QUAD{4, 128, Dim, BuildType::FAST_BUILD, LinkType::QUADRATIC_CURVE, 16, 8, false};
        constexpr batchRobotConfig BM_IAS{4, 1024, Dim, BuildType::COMPACT, LinkType::MESH, true};
        constexpr batchRobotConfig BM_RAY{4, 1024, Dim, BuildType::NONE, LinkType::RAY_ONLINE, true};
        constexpr batchRobotConfig BM_LIN8{16, 16, Dim, BuildType::FAST_BUILD, LinkType::LINEAR_CURVE, 8, 8, false};
        constexpr batchRobotConfig BM_LIN16{16, 16, Dim, BuildType::FAST_BUILD, LinkType::LINEAR_CURVE, 16, 16, false};
        constexpr batchRobotConfig BM_QUAD4_8{16, 16, Dim, BuildType::FAST_TRACE, LinkType::QUADRATIC_CURVE, 8, 4, false};
        constexpr batchRobotConfig BM_QUAD8_16{16, 16, Dim, BuildType::FAST_TRACE, LinkType::QUADRATIC_CURVE, 16, 8, false};
        constexpr batchRobotConfig BM_CMESH{16, 1024, Dim, BuildType::COMPACT, LinkType::MESH, true};
        constexpr batchRobotConfig BM_CRAY{16, 1024, Dim, BuildType::NONE, LinkType::RAY_ONLINE, true};

        constexpr std::size_t kCurvePoseCap = 4096 * 32; // BM_MESH / BM_RAY<32> of the curve benchmark

        struct Entry {
            OldProfile profile;
            std::unique_ptr<OldRunner> (*make)(const OldProfile&, OptixDeviceContext, const OldSettings&);
        };
        template <batchRobotConfig CFG>
        std::unique_ptr<OldRunner> make(const OldProfile& p, OptixDeviceContext c, const OldSettings& s) {
            return std::make_unique<Runner<CFG>>(p, c, s);
        }

        const std::vector<Entry>& entries() {
            static const std::vector<Entry> e = {
                {{"reg_ias", "ccRegression_ias", OldMode::Ias, 4, 128, 1, 0, OldBuild::Compact, true, 4096}, make<REG_IAS>},
                {{"reg_ray", "ccRegression_ray", OldMode::Ray, 4, 128, 1, 0, OldBuild::None, true, 4096}, make<REG_RAY>},
                {{"reg_linear", "ccRegression_linear", OldMode::Linear, 4, 128, 16, 16, OldBuild::FastBuild, false, 4096},
                    make<REG_LINEAR>},
                {{"reg_quad", "ccRegression_quad", OldMode::Quad, 4, 128, 16, 8, OldBuild::FastBuild, false, 4096},
                    make<REG_QUAD>},
                {{"bm_ias", "benchmark<Scene> BM_IAS", OldMode::Ias, 4, 1024, 1, 0, OldBuild::Compact, true, 8196},
                    make<BM_IAS>},
                {{"bm_ray", "benchmark<Scene> BM_RAY", OldMode::Ray, 4, 1024, 1, 0, OldBuild::None, true, 8196},
                    make<BM_RAY>},
                {{"bm_linear_8", "BM_CURVE_<Scene> BM_LINEAR<8,8,16>", OldMode::Linear, 16, 16, 8, 8, OldBuild::FastBuild,
                     false, 4096},
                    make<BM_LIN8>},
                {{"bm_linear_16", "BM_CURVE_<Scene> BM_LINEAR<16,16,16>", OldMode::Linear, 16, 16, 16, 16,
                     OldBuild::FastBuild, false, 4096},
                    make<BM_LIN16>},
                {{"bm_quad_4_8", "BM_CURVE_<Scene> BM_QUAD<4,8,16>", OldMode::Quad, 16, 16, 8, 4, OldBuild::FastTrace,
                     false, 4096},
                    make<BM_QUAD4_8>},
                {{"bm_quad_8_16", "BM_CURVE_<Scene> BM_QUAD<8,16,16>", OldMode::Quad, 16, 16, 16, 8, OldBuild::FastTrace,
                     false, 4096},
                    make<BM_QUAD8_16>},
                {{"bm_curve_mesh", "BM_CURVE_<Scene> BM_MESH<16|32>", OldMode::Ias, 16, 1024, 1, 0, OldBuild::Compact,
                     true, kCurvePoseCap},
                    make<BM_CMESH>},
                {{"bm_curve_ray", "BM_CURVE_<Scene> BM_RAY<16|32>", OldMode::Ray, 16, 1024, 1, 0, OldBuild::None, true,
                     kCurvePoseCap},
                    make<BM_CRAY>},
            };
            return e;
        }
    } // namespace

    const std::vector<OldProfile>& oldProfiles() {
        static const std::vector<OldProfile> v = [] {
            std::vector<OldProfile> out;
            for (const auto& e : entries()) {
                out.push_back(e.profile);
            }
            return out;
        }();
        return v;
    }

    const OldProfile* findOldProfile(const std::string& name) {
        for (const auto& p : oldProfiles()) {
            if (p.name == name) {
                return &p;
            }
        }
        return nullptr;
    }

    std::unique_ptr<OldRunner> createOldRunner(const std::string& name, void* optixContext, const OldSettings& s) {
        for (const auto& e : entries()) {
            if (e.profile.name == name) {
                return e.make(e.profile, static_cast<OptixDeviceContext>(optixContext), s);
            }
        }
        throw std::runtime_error("unknown old-stack profile '" + name + "'");
    }

} // namespace rtcd::sbs
