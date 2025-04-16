#define PRINTIT
#include <Robot/batchIASRobot.h>
#include <Robot/models/panda.h>
#include <Utils/Test/testUtils.h>
#include <random>

inline constexpr size_t NSTREAM   = 2;
inline constexpr size_t BATCHSIZE = 1024;
inline constexpr batchRobotConfig batchConfig{
    NSTREAM, BATCHSIZE, Dim, BuildType::COMPACT, LinkType::SPHERES, false, true};

using namespace RTCD;
int main() {

    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();

    auto r          = std::make_shared<robot<Dim>>(Config);
    auto batchRobot = batchIASRobot<batchConfig>(r, streams, optixContext);


    // Prepare Trajectory
    std::vector<std::array<float, Dim>> traj(4096);
    randomMotions(4096, traj, lowerBound, upperBound, r->name);

    CUDABuffer trajBuffer;
    trajBuffer.alloc_and_upload(traj);

    batchRobot.fkine(trajBuffer.d_pointer(), 0);


    std::vector<float> gTFs;
    batchRobot.downloadTF(gTFs, 0);
    writeBin(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/robotFkineTest.bin"), gTFs);

    CUDA_SYNC_CHECK();
    return 0;
}
