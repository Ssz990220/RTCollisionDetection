// GPU tests of rtcd::Scene (label new_gpu), new stack only:
//   * construction of the medium shelf scene (configs equivalent of RTCD/CollisionScenes/Scene/shelf.h): obstacle
//     infos, edge-ray buffers (both sets) and OBBs on the device equal what the host meshes give;
//   * IAS / GAS / SBT sanity: random segments traced against Scene::traversable() with a small test pipeline
//     (scene_trace_program.cu) agree with a brute-force host intersection of the world meshes, and instance o
//     uses SBT record o;
//   * constructor errors.

#ifndef NOMINMAX
#define NOMINMAX // optix_stubs.h includes <windows.h>
#endif
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/scene.h>

#include "scene_trace_params.h"
#include <test_scene_trace_ir_embedded.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

using namespace rtcd;

namespace {

    std::shared_ptr<Context> sharedContext() {
        static std::shared_ptr<Context> ctx = Context::create();
        return ctx;
    }

    SceneDesc shelfDesc() {
        struct O {
            const char* file;
            float x, y, z;
        };
        const O obs[] = {{"models/curobo/shelves.obj", 0.45f, 0, 0.4f}, {"models/curobo/bin.obj", 0, -0.6f, 0},
            {"models/curobo/bin.obj", 0, 0.6f, 0}, {"models/curobo/Bob.obj", -0.4f, 0, 0},
            {"models/curobo/Cow.obj", -0.4f, 0, 0}, {"models/curobo/Fish.obj", -0.4f, 0, 0},
            {"models/curobo/Sheep.obj", -0.4f, 0, 0}, {"models/curobo/Snakeboard.obj", -0.4f, 0, 0}};
        SceneDesc d;
        d.name = "shelf";
        for (const O& o : obs) {
            ObstacleDesc ob;
            ob.mesh = o.file;
            ob.pose = Transform::translation(o.x, o.y, o.z);
            d.obstacles.push_back(ob);
        }
        return d;
    }

    const Scene& shelfScene() {
        static Scene scene(sharedContext(), shelfDesc());
        return scene;
    }

    template <typename T>
    std::vector<T> download(DeviceSpan<const T> s) {
        std::vector<T> h(s.count);
        if (s.count) {
            EXPECT_EQ(cudaMemcpy(h.data(), reinterpret_cast<const void*>(s.ptr), s.bytes(), cudaMemcpyDeviceToHost), cudaSuccess);
        }
        return h;
    }

    void checkEdgeSet(const Scene& scene, EdgeSet set) {
        const EdgeRays& rays = scene.edgeRays(set);
        ASSERT_EQ(rays.firstRay.size(), scene.numObstacles());
        ASSERT_EQ(rays.rayCount.size(), scene.numObstacles());
        ASSERT_EQ(rays.vertices.count, 6u * rays.count);
        const std::vector<float> dev = download(rays.vertices);
        std::uint32_t first = 0;
        for (std::uint32_t o = 0; o < scene.numObstacles(); ++o) {
            const TriangleMesh& m          = scene.meshes()[o];
            const std::vector<Edge>& edges = set == EdgeSet::Unique ? m.uniqueEdges : m.loopEdges;
            EXPECT_EQ(rays.firstRay[o], first);
            ASSERT_EQ(rays.rayCount[o], edges.size());
            for (std::size_t e = 0; e < edges.size(); ++e) {
                const float* r = dev.data() + 6 * (first + e);
                const Vec3& a  = m.vertices[edges[e][0]];
                const Vec3& b  = m.vertices[edges[e][1]];
                ASSERT_EQ(std::memcmp(r, a.data(), 12), 0) << "obstacle " << o << " ray " << e;
                ASSERT_EQ(std::memcmp(r + 3, b.data(), 12), 0) << "obstacle " << o << " ray " << e;
            }
            first += rays.rayCount[o];
        }
        EXPECT_EQ(rays.count, first);
    }

    // ---- minimal OptiX pipeline around scene_trace_program.cu ----
    struct alignas(16) Record {
        char header[32];
        unsigned int data;
    };
    static_assert(sizeof(Record) == 48);

    class TracePipeline {
    public:
        explicit TracePipeline(const Context& ctx, std::uint32_t numHitRecords) {
            char log[4096];
            size_t logSize = sizeof(log);
            OptixModuleCompileOptions mco = {};
            OptixPipelineCompileOptions pco = {};
            pco.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pco.numPayloadValues                 = 2;
            pco.numAttributeValues               = 2;
            pco.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
            pco.pipelineLaunchParamsVariableName = "rtcdLaunchParams";
            pco.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            RTCD_OPTIX_CHECK_LOG(optixModuleCreate(ctx.optixContext(), &mco, &pco,
                                     reinterpret_cast<const char*>(rtcd::embedded::test_scene_trace_ir_scene_trace_program),
                                     rtcd::embedded::test_scene_trace_ir_scene_trace_program_size, log, &logSize, &module_),
                log);

            OptixProgramGroupDesc desc[3] = {};
            desc[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            desc[0].raygen.module            = module_;
            desc[0].raygen.entryFunctionName = "__raygen__trace";
            desc[1].kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc[1].miss.module              = module_;
            desc[1].miss.entryFunctionName   = "__miss__trace";
            desc[2].kind                             = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[2].hitgroup.moduleCH            = module_;
            desc[2].hitgroup.entryFunctionNameCH = "__closesthit__trace";
            OptixProgramGroupOptions pgo         = {};
            logSize                              = sizeof(log);
            RTCD_OPTIX_CHECK_LOG(optixProgramGroupCreate(ctx.optixContext(), desc, 3, &pgo, log, &logSize, groups_), log);

            OptixPipelineLinkOptions plo = {};
            plo.maxTraceDepth            = 1;
            logSize                      = sizeof(log);
            RTCD_OPTIX_CHECK_LOG(
                optixPipelineCreate(ctx.optixContext(), &pco, &plo, groups_, 3, log, &logSize, &pipeline_), log);
            RTCD_OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, 0, 0, 1024, 2));

            std::vector<Record> rg(1), ms(1), hg(numHitRecords);
            RTCD_OPTIX_CHECK(optixSbtRecordPackHeader(groups_[0], &rg[0]));
            RTCD_OPTIX_CHECK(optixSbtRecordPackHeader(groups_[1], &ms[0]));
            for (std::uint32_t i = 0; i < numHitRecords; ++i) {
                RTCD_OPTIX_CHECK(optixSbtRecordPackHeader(groups_[2], &hg[i]));
                hg[i].data = i;
            }
            raygen_ = DeviceBuffer::fromHost(std::span<const Record>(rg));
            miss_   = DeviceBuffer::fromHost(std::span<const Record>(ms));
            hit_    = DeviceBuffer::fromHost(std::span<const Record>(hg));
            sbt_.raygenRecord                = raygen_.ptr();
            sbt_.missRecordBase              = miss_.ptr();
            sbt_.missRecordStrideInBytes     = sizeof(Record);
            sbt_.missRecordCount             = 1;
            sbt_.hitgroupRecordBase          = hit_.ptr();
            sbt_.hitgroupRecordStrideInBytes = sizeof(Record);
            sbt_.hitgroupRecordCount         = numHitRecords;
        }
        ~TracePipeline() {
            optixPipelineDestroy(pipeline_);
            for (auto g : groups_) {
                optixProgramGroupDestroy(g);
            }
            optixModuleDestroy(module_);
        }
        void launch(const SceneTraceParams& p) {
            DeviceBuffer params = DeviceBuffer::fromHost(std::span<const SceneTraceParams>(&p, 1));
            RTCD_OPTIX_CHECK(optixLaunch(pipeline_, nullptr, params.ptr(), sizeof(p), &sbt_, p.count, 1, 1));
            RTCD_CUDA_CHECK(cudaDeviceSynchronize());
        }

    private:
        OptixModule module_            = nullptr;
        OptixProgramGroup groups_[3]   = {};
        OptixPipeline pipeline_        = nullptr;
        DeviceBuffer raygen_, miss_, hit_;
        OptixShaderBindingTable sbt_ = {};
    };

    // ---- brute-force host reference ----
    struct HostHit {
        bool ambiguous = false;     // a hit close to a triangle edge or to a segment end: skipped
        int nearest    = -1;        // obstacle of the nearest hit
        double tNearest = 2.0;
        double tOther   = 2.0;      // nearest hit on a DIFFERENT obstacle
    };

    HostHit hostTrace(const Scene& scene, const float* ray) {
        const double o[3] = {ray[0], ray[1], ray[2]};
        const double d[3] = {ray[3] - ray[0], ray[4] - ray[1], ray[5] - ray[2]};
        constexpr double eps = 1e-5;
        HostHit h;
        for (std::uint32_t ob = 0; ob < scene.numObstacles(); ++ob) {
            const TriangleMesh& m = scene.meshes()[ob];
            for (const auto& t : m.triangles) {
                const Vec3& a = m.vertices[t[0]];
                const Vec3& b = m.vertices[t[1]];
                const Vec3& c = m.vertices[t[2]];
                const double e1[3] = {b[0] - double(a[0]), b[1] - double(a[1]), b[2] - double(a[2])};
                const double e2[3] = {c[0] - double(a[0]), c[1] - double(a[1]), c[2] - double(a[2])};
                const double p[3]  = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
                const double det   = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
                if (std::abs(det) < 1e-14) {
                    continue;
                }
                const double inv  = 1.0 / det;
                const double s[3] = {o[0] - a[0], o[1] - a[1], o[2] - a[2]};
                const double u    = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
                const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
                const double v    = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
                const double tt   = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
                const double w    = 1.0 - u - v;
                const double minB = std::min({u, v, w});
                if (minB < -eps || tt < -eps || tt > 1 + eps) {
                    continue;
                }
                if (minB < eps || tt < eps || tt > 1 - eps) {
                    h.ambiguous = true;
                    continue;
                }
                if (tt < h.tNearest) {
                    if (h.nearest != static_cast<int>(ob)) {
                        h.tOther = h.tNearest;
                    }
                    h.tNearest = tt;
                    h.nearest  = static_cast<int>(ob);
                } else if (static_cast<int>(ob) != h.nearest && tt < h.tOther) {
                    h.tOther = tt;
                }
            }
        }
        return h;
    }

} // namespace

TEST(SceneGpu, ShelfInfosEdgesAndObbs) {
    const Scene& scene = shelfScene();
    ASSERT_EQ(scene.numObstacles(), 8u);
    EXPECT_EQ(scene.desc().name, "shelf");
    EXPECT_EQ(scene.context(), sharedContext());
    EXPECT_NE(scene.traversable(), 0u);
    EXPECT_EQ(scene.obstacles()[0].name, "shelves"); // default: mesh file stem
    EXPECT_EQ(scene.obstacles()[1].name, "bin");
    ASSERT_EQ(scene.meshes().size(), 8u);
    for (std::uint32_t o = 0; o < 8; ++o) {
        const ObstacleInfo& info = scene.obstacles()[o];
        EXPECT_EQ(info.numVertices, scene.meshes()[o].vertices.size());
        EXPECT_EQ(info.numTriangles, scene.meshes()[o].triangles.size());
        EXPECT_TRUE(info.watertight);
    }
    // world-space vertices: bin2 = bin1 shifted by +1.2 in y
    EXPECT_EQ(scene.meshes()[1].vertices.size(), scene.meshes()[2].vertices.size());

    EXPECT_TRUE(scene.hasEdgeSet(EdgeSet::Unique));
    EXPECT_TRUE(scene.hasEdgeSet(EdgeSet::Loop));
    checkEdgeSet(scene, EdgeSet::Unique);
    checkEdgeSet(scene, EdgeSet::Loop);

    const std::vector<Obb> obbs = download(scene.obbs());
    ASSERT_EQ(obbs.size(), 8u);
    for (std::uint32_t o = 0; o < 8; ++o) {
        const Obb ref = fitObb(scene.meshes()[o].vertices);
        EXPECT_EQ(std::memcmp(&obbs[o], &ref, sizeof(Obb)), 0) << o;
        EXPECT_EQ(std::memcmp(&scene.obstacles()[o].obb, &ref, sizeof(Obb)), 0) << o;
    }
}

TEST(SceneGpu, TraceAgreesWithHostIntersection) {
    const Scene& scene = shelfScene();
    constexpr std::uint32_t n = 4096;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(-1.0f, 1.2f), uy(-1.0f, 1.0f), uz(-0.2f, 1.4f), ul(-0.4f, 0.4f);
    std::vector<float> rays(6 * n);
    for (std::uint32_t i = 0; i < n; ++i) {
        float* r = rays.data() + 6 * i;
        r[0]     = ux(rng);
        r[1]     = uy(rng);
        r[2]     = uz(rng);
        r[3]     = r[0] + ul(rng);
        r[4]     = r[1] + ul(rng);
        r[5]     = r[2] + ul(rng);
    }
    DeviceBuffer dRays = DeviceBuffer::fromHost(std::span<const float>(rays));
    DeviceBuffer dSbt(n * sizeof(unsigned int)), dInst(n * sizeof(unsigned int));

    TracePipeline pipeline(*sharedContext(), scene.numObstacles());
    SceneTraceParams p{};
    p.handle   = scene.traversable();
    p.rays     = reinterpret_cast<const float*>(dRays.ptr());
    p.count    = n;
    p.sbtData  = reinterpret_cast<unsigned int*>(dSbt.ptr());
    p.instance = reinterpret_cast<unsigned int*>(dInst.ptr());
    pipeline.launch(p);

    std::vector<unsigned int> sbt(n), inst(n);
    dSbt.download(sbt.data(), n * sizeof(unsigned int));
    dInst.download(inst.data(), n * sizeof(unsigned int));

    std::uint32_t hits = 0, checked = 0, nearestChecked = 0, hitMismatch = 0, nearestMismatch = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(sbt[i], inst[i]) << "ray " << i << ": instance o must use SBT record o";
        const HostHit h = hostTrace(scene, rays.data() + 6 * i);
        if (h.ambiguous) {
            continue;
        }
        ++checked;
        const bool hostHit = h.nearest >= 0;
        hits += hostHit;
        if (hostHit != (inst[i] != 0)) {
            if (hitMismatch++ < 5) {
                ADD_FAILURE() << "ray " << i << ": host " << hostHit << " device " << (inst[i] != 0);
            }
            continue;
        }
        if (hostHit && h.tOther - h.tNearest > 1e-4) {
            ++nearestChecked;
            if (static_cast<int>(inst[i]) - 1 != h.nearest && nearestMismatch++ < 5) {
                ADD_FAILURE() << "ray " << i << ": nearest obstacle host " << h.nearest << " device " << inst[i] - 1;
            }
        }
    }
    EXPECT_EQ(hitMismatch, 0u);
    EXPECT_EQ(nearestMismatch, 0u);
    EXPECT_GT(checked, n * 9 / 10);
    EXPECT_GT(hits, n / 20) << "too few hits for a meaningful test";
    std::cout << "rays " << n << ", checked " << checked << ", host hits " << hits << ", nearest checked "
              << nearestChecked << "\n";
}

TEST(SceneGpu, ScaleAndPoseApplied) {
    SceneDesc d;
    d.name = "scaled";
    ObstacleDesc ob;
    ob.mesh  = "models/curobo/Sheep.obj";
    ob.name  = "sheep";
    ob.pose  = Transform::fromXyzRpy(0.1, 0.2, 0.3, 0, 0, 0.5);
    ob.scale = 2.0f;
    d.obstacles.push_back(ob);
    const Scene s(sharedContext(), d);
    const TriangleMesh ref = loadObjMesh(ob.mesh, ob.pose, ob.scale);
    ASSERT_EQ(s.meshes()[0].vertices, ref.vertices);
    EXPECT_EQ(s.obstacles()[0].name, "sheep");
    checkEdgeSet(s, EdgeSet::Unique);
}

TEST(SceneGpu, ConstructorErrors) {
    SceneDesc empty;
    empty.name = "empty";
    EXPECT_THROW(Scene(nullptr, shelfDesc()), InvalidArgument);
    EXPECT_THROW(Scene(sharedContext(), empty), InvalidArgument);

    SceneDesc missing;
    missing.obstacles.push_back(ObstacleDesc{"nope", "models/curobo/does_not_exist.obj", Transform{}, 1.0f});
    EXPECT_THROW(Scene(sharedContext(), missing), IoError);

    SceneDesc badScale;
    badScale.obstacles.push_back(ObstacleDesc{"bin", "models/curobo/bin.obj", Transform{}, 0.0f});
    EXPECT_THROW(Scene(sharedContext(), badScale), InvalidArgument);
}
