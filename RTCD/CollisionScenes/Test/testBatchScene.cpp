#include <CollisionScenes/Scene/shelf.h>
#include <CollisionScenes/batchScene.h>
#include <Utils/Test/testUtils.h>
#include <Utils/cuUtils.cuh>
#include <algorithm>
#include <random>

inline constexpr size_t NSTREAM = 4;

inline constexpr batchSceneConfig CFG(true, 4);

using namespace RTCD;
int main() {
    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

    std::shared_ptr<scene> s = buildSharedScene();
    batchScene<CFG> ss(s, streams, optixContext);


#if defined(_DEBUG) || defined(DEBUG)
    CUDA_SYNC_CHECK();
    // download the OBBs for visual check
    writeBin(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/obsOBBs.bin"), s->getOBBs());
    std::cout << "OBBs saved to " << CONCAT_PATHS(PROJECT_BASE_DIR, "/data/obsOBBs.bin") << std::endl;
#endif

    // Initialize random number generation
    std::vector<uint> mask(ss.getNumObstacles());
    std::random_device rd; // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> distrib(0, 4); // Define the range [0, 3]

    std::generate(mask.begin(), mask.end(), [&]() { return distrib(gen); });

    CUDABuffer dMask;
    dMask.alloc_and_upload(mask);

    ss.updateASWithMask(dMask.d_pointer(), 0);

    CUDA_SYNC_CHECK();

    std::vector<uint> vertCnts = s->getObsVertCnt();
    std::vector<float3> verts  = s->getSgVerts();

    std::cout << "There are " << verts.size() << " verts\n";

    std::vector<float3> selectedVerts;

    int start = 0;
    for (int i = 0; i < ss.getNumObstacles(); i++) {
        if (mask[i] > 0) {
            selectedVerts.insert(selectedVerts.end(), verts.begin() + start, verts.begin() + start + vertCnts[i]);
        }
        start += vertCnts[i];
    }


    std::cout << "There are " << selectedVerts.size() << " selected verts\n";

    ss.updateEdgeWithMask(dMask.d_pointer(), 0);
    CUDA_SYNC_CHECK();
    std::vector<meshRayInfo> rayInfo = ss.getRayInfo(0);

    int rayCount = 0;
    for (int i = 0; i < rayInfo.size(); i++) {
        rayCount += rayInfo[i].nRays;
    }

    std::vector<float3> rays(rayCount * 2);
    start = 0;
    for (int i = 0; i < rayInfo.size(); i++) {
        cudaMemcpy(rays.data() + start, (void*) rayInfo[i].meshRays, rayInfo[i].nRays * sizeof(float3) * 2,
            cudaMemcpyDeviceToHost);
        start += rayInfo[i].nRays * 2;
    }
    std::cout << "There are " << rays.size() << " rays\n";
    CUDA_SYNC_CHECK();

    if (std::equal(selectedVerts.begin(), selectedVerts.end(), rays.begin(),
            [](float3& a, float3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; })) {
        std::cout << "Selected verts are the same\n";
    } else {
        std::cout << "Selected verts are not the same\n";
    }

    return 0;
}
