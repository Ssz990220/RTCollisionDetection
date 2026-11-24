// Parity of rtcd::Scene with the old RTCD::scene (RTCD/CollisionScenes/scene.h, obstacle.h) on the three shelf
// scenes (shelf.h, denseShelf.h, shelfSimple.h): the edge-ray buffers of both edge sets (old obstacle(..., loop)
// singleVertices, concatenated by scene::addObstacle and uploaded by scene::uploadToDevice), the per-obstacle
// ray counts / offsets (old obsVertCnt / whereInEdge, in vertices = 2 per ray) and the obstacle OBBs (old
// scene::getdOBBs), compared bit for bit on the device buffers. Label new_gpu (both scenes upload).
// Includes OLD-STACK headers: DELETE THIS TEST TOGETHER WITH THE OLD STACK. With RTCD_WRITE_REFERENCE=<dir> it
// writes tests/data/reference/scenes.txt from the old scenes (the stored reference of test_scene_reference.cpp).

#ifndef NOMINMAX
#define NOMINMAX // optix_stubs.h (via the old headers) includes <windows.h>
#endif
#include <rtcd/context.h>
#include <rtcd/scene.h>

// old stack (reference). The OptiX function table comes from rtcd_core (its own object file).
#include <CollisionScenes/scene.h>

#include <support/reference.h>

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>

namespace {

    struct ObstacleRef {
        const char* file;
        float x, y, z;
    };

    const std::vector<ObstacleRef> kShelf = {{"curobo/shelves.obj", 0.45f, 0, 0.4f}, {"curobo/bin.obj", 0, -0.6f, 0},
        {"curobo/bin.obj", 0, 0.6f, 0}, {"curobo/Bob.obj", -0.4f, 0, 0}, {"curobo/Cow.obj", -0.4f, 0, 0},
        {"curobo/Fish.obj", -0.4f, 0, 0}, {"curobo/Sheep.obj", -0.4f, 0, 0}, {"curobo/Snakeboard.obj", -0.4f, 0, 0}};
    const std::vector<ObstacleRef> kDenseShelf = {{"curobo/shelves.obj", 0.45f, 0, 0.4f},
        {"curobo/bin_dense.obj", 0, -0.6f, 0}, {"curobo/bin_dense.obj", 0, 0.6f, 0}, {"curobo/Bob.obj", -0.4f, 0, 0},
        {"curobo/Face1.obj", -0.4f, 0, 0}, {"curobo/Face2.obj", -0.4f, 0, 0}, {"curobo/Face3.obj", -0.4f, 0, 0},
        {"curobo/Cow.obj", -0.4f, 0, 0}, {"curobo/Fish.obj", -0.4f, 0, 0}, {"curobo/Sheep.obj", -0.4f, 0, 0},
        {"curobo/Snakeboard.obj", -0.4f, 0, 0}};
    const std::vector<ObstacleRef> kShelfSimple = {
        {"curobo/shelves.obj", 0.45f, 0, 0.4f}, {"curobo/bin.obj", 0, -0.6f, 0}, {"curobo/bin.obj", 0, 0.6f, 0}};

    std::shared_ptr<rtcd::Context> sharedContext() {
        static std::shared_ptr<rtcd::Context> ctx = rtcd::Context::create();
        return ctx;
    }

    // Old buildSharedScene(loop) for an obstacle list (sceneBuilder.h + the scene headers' addObj).
    std::unique_ptr<RTCD::scene> buildOld(const std::vector<ObstacleRef>& list, bool loop) {
        auto s = std::make_unique<RTCD::scene>();
        for (const ObstacleRef& ob : list) {
            Eigen::Affine3f pose = Eigen::Affine3f::Identity();
            pose.translate(Eigen::Vector3f(ob.x, ob.y, ob.z));
            s->addObstacle(std::make_unique<RTCD::obstacle>(std::string("models/") + ob.file, pose, 1.0f, loop));
        }
        s->uploadToDevice();
        return s;
    }

    rtcd::SceneDesc newDesc(const std::vector<ObstacleRef>& list) {
        rtcd::SceneDesc d;
        for (const ObstacleRef& ob : list) {
            d.obstacles.push_back(
                rtcd::ObstacleDesc{"", std::string("models/") + ob.file, rtcd::Transform::translation(ob.x, ob.y, ob.z), 1.0f});
        }
        return d;
    }

    template <typename T>
    std::vector<T> downloadBytes(CUdeviceptr p, std::size_t count) {
        std::vector<T> h(count);
        EXPECT_EQ(cudaMemcpy(h.data(), reinterpret_cast<const void*>(p), count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }

    void compareEdgeSet(const RTCD::scene& old, const rtcd::Scene& now, rtcd::EdgeSet set) {
        SCOPED_TRACE(set == rtcd::EdgeSet::Unique ? "unique" : "loop");
        const rtcd::EdgeRays& rays = now.edgeRays(set);
        ASSERT_EQ(old.getSgRayCnt(), rays.count);
        // device buffers, byte for byte
        const auto oldDev = downloadBytes<float>(old.getdSgVerts(), 6 * rays.count);
        const auto newDev = downloadBytes<float>(rays.vertices.ptr, 6 * rays.count);
        EXPECT_EQ(std::memcmp(oldDev.data(), newDev.data(), oldDev.size() * sizeof(float)), 0);
        // per-obstacle counts / offsets: old in vertices (2 per ray)
        const std::vector<uint> cnt = old.getObsVertCnt();
        ASSERT_EQ(cnt.size(), now.numObstacles());
        std::vector<uint> where(cnt.size());
        std::exclusive_scan(cnt.begin(), cnt.end(), where.begin(), 0u);
        const auto oldWhere = downloadBytes<uint>(old.getWhereInEdgePtr(), cnt.size());
        for (std::size_t o = 0; o < cnt.size(); ++o) {
            EXPECT_EQ(cnt[o], 2 * rays.rayCount[o]) << o;
            EXPECT_EQ(oldWhere[o], 2 * rays.firstRay[o]) << o;
            EXPECT_EQ(where[o], oldWhere[o]) << o;
        }
    }

    void compareScene(const std::vector<ObstacleRef>& list) {
        const auto ctx = sharedContext();
        const rtcd::Scene now(ctx, newDesc(list));
        ASSERT_EQ(now.numObstacles(), list.size());
        ASSERT_TRUE(now.hasEdgeSet(rtcd::EdgeSet::Loop));
        for (const bool loop : {false, true}) {
            const auto old = buildOld(list, loop);
            ASSERT_EQ(old->getNumObstacles(), now.numObstacles());
            compareEdgeSet(*old, now, loop ? rtcd::EdgeSet::Loop : rtcd::EdgeSet::Unique);

            static_assert(sizeof(RTCD::OBB) == sizeof(rtcd::Obb));
            const auto oldObbs = downloadBytes<RTCD::OBB>(old->getdOBBs(), now.numObstacles());
            const auto newObbs = downloadBytes<rtcd::Obb>(now.obbs().ptr, now.numObstacles());
            EXPECT_EQ(std::memcmp(oldObbs.data(), newObbs.data(), newObbs.size() * sizeof(rtcd::Obb)), 0) << "OBBs";
            for (std::uint32_t o = 0; o < now.numObstacles(); ++o) {
                EXPECT_EQ(std::memcmp(&old->getOBBs()[o], &now.obstacles()[o].obb, sizeof(rtcd::Obb)), 0) << o;
            }
        }
    }

} // namespace

TEST(SceneParity, Shelf) {
    compareScene(kShelf);
}
TEST(SceneParity, DenseShelf) {
    compareScene(kDenseShelf);
}
TEST(SceneParity, ShelfSimple) {
    compareScene(kShelfSimple);
}

// GENERATOR (runs only with RTCD_WRITE_REFERENCE=<dir>): scenes.txt from the OLD RTCD::scene.
TEST(SceneParity, WriteStoredReferences) {
    namespace sup  = rtcd::support;
    const auto dir = sup::referenceWriteDir();
    if (!dir) {
        GTEST_SKIP() << "set RTCD_WRITE_REFERENCE=<dir> to write the stored references";
    }
    sup::ReferenceWriter w(*dir / "scenes.txt",
        "Edge-ray buffers and obstacle OBBs of the OLD RTCD::scene (RTCD/CollisionScenes) for the three shelf scenes,\n"
        "written by tests/scene/test_scene_parity.cpp (RTCD_WRITE_REFERENCE). Device buffers as uploaded:\n"
        "<scene>.<unique|loop>.rays <rays> <FNV-1a of the 6 floats (origin, target) per ray>; .ray_counts and\n"
        ".first_ray per obstacle in rays (old obsVertCnt / whereInEdge halved); <scene>.obbs <n> <FNV-1a of the OBBs>.");
    const std::pair<const char*, const std::vector<ObstacleRef>*> scenes[] = {
        {"shelf", &kShelf}, {"shelf_dense", &kDenseShelf}, {"shelf_simple", &kShelfSimple}};
    for (const auto& [name, list] : scenes) {
        sharedContext()->makeCurrent();
        for (const bool loop : {false, true}) {
            const auto old        = buildOld(*list, loop);
            const std::string key = std::string(name) + (loop ? ".loop" : ".unique");
            const std::size_t n   = old->getSgRayCnt();
            const auto rays       = downloadBytes<float>(old->getdSgVerts(), 6 * n);
            w.putHash(key + ".rays", n, sup::fnv1a(rays));
            std::vector<std::uint32_t> counts, first;
            const auto where = downloadBytes<uint>(old->getWhereInEdgePtr(), old->getObsVertCnt().size());
            for (std::size_t o = 0; o < where.size(); ++o) {
                counts.push_back(old->getObsVertCnt()[o] / 2);
                first.push_back(where[o] / 2);
            }
            w.putInts(key + ".ray_counts", counts);
            w.putInts(key + ".first_ray", first);
            if (!loop) {
                const auto obbs = downloadBytes<RTCD::OBB>(old->getdOBBs(), old->getNumObstacles());
                w.putHash(std::string(name) + ".obbs", obbs.size(), sup::fnv1a(obbs));
            }
        }
    }
    w.save();
}