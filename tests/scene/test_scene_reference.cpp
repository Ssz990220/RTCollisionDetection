// rtcd::Scene on the three shelf scenes (configs/scenes/shelf*.json) vs the stored reference
// tests/data/reference/scenes.txt, written from the OLD RTCD::scene before the old stack was removed: the device
// edge-ray buffers of both edge sets (count + FNV-1a of the 6 floats per ray), the per-obstacle ray counts / first
// rays, and the device obstacle OBBs (count + FNV-1a). Exact. Label new_gpu (the scene uploads).
#include <rtcd/context.h>
#include <rtcd/json_config.h>
#include <rtcd/scene.h>

#include <support/paths.h>
#include <support/reference.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace sup = rtcd::support;

namespace {

    template <typename T>
    std::vector<T> download(rtcd::DevicePtr p, std::size_t count) {
        std::vector<T> h(count);
        EXPECT_EQ(cudaMemcpy(h.data(), reinterpret_cast<const void*>(p), count * sizeof(T), cudaMemcpyDeviceToHost),
            cudaSuccess);
        return h;
    }

    void compareScene(const std::string& name) {
        static const sup::ReferenceFile ref = sup::ReferenceFile::load(sup::referenceDir() / "scenes.txt");
        static const auto ctx               = rtcd::Context::create();
        const rtcd::Scene scene(ctx, rtcd::loadSceneDesc(sup::configsDir() / "scenes" / (name + ".json")));
        for (const auto set : {rtcd::EdgeSet::Unique, rtcd::EdgeSet::Loop}) {
            const std::string key = name + (set == rtcd::EdgeSet::Loop ? ".loop" : ".unique");
            SCOPED_TRACE(key);
            ASSERT_TRUE(scene.hasEdgeSet(set));
            const rtcd::EdgeRays& rays = scene.edgeRays(set);
            ASSERT_EQ(rays.count, ref.integer(key + ".rays", 0));
            const auto v = download<float>(rays.vertices.ptr, 6 * std::size_t(rays.count));
            EXPECT_EQ(sup::hex64(sup::fnv1a(v)), sup::hex64(ref.hash(key + ".rays", 1))) << "edge-ray buffer";
            const auto counts = ref.integers(key + ".ray_counts");
            const auto first  = ref.integers(key + ".first_ray");
            ASSERT_EQ(counts.size(), scene.numObstacles());
            ASSERT_EQ(first.size(), scene.numObstacles());
            for (std::uint32_t o = 0; o < scene.numObstacles(); ++o) {
                EXPECT_EQ(rays.rayCount[o], counts[o]) << "obstacle " << o;
                EXPECT_EQ(rays.firstRay[o], first[o]) << "obstacle " << o;
            }
        }
        ASSERT_EQ(scene.numObstacles(), ref.integer(name + ".obbs", 0));
        const auto obbs = download<rtcd::Obb>(scene.obbs().ptr, scene.numObstacles());
        EXPECT_EQ(sup::hex64(sup::fnv1a(obbs)), sup::hex64(ref.hash(name + ".obbs", 1))) << "obstacle OBBs";
        for (std::uint32_t o = 0; o < scene.numObstacles(); ++o) {
            EXPECT_EQ(std::memcmp(&obbs[o], &scene.obstacles()[o].obb, sizeof(rtcd::Obb)), 0) << "host copy, " << o;
        }
    }

} // namespace

TEST(SceneReference, Shelf) {
    compareScene("shelf");
}
TEST(SceneReference, DenseShelf) {
    compareScene("shelf_dense");
}
TEST(SceneReference, ShelfSimple) {
    compareScene("shelf_simple");
}
