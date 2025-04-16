#include <Robot/batchGASRobot.h>
#include <Robot/models/panda.h>
#include <Utils/Test/testUtils.h>

inline constexpr batchRobotConfig batchConfig{2, 16, Dim, BuildType::FAST_TRACE, LinkType::SPHERE_GAS};

using namespace RTCD;
int main() {

    auto [cudaContext, optixContext, streams] = createContextStream<batchConfig.NSTREAM>();

    std::shared_ptr<robot<Dim>> Robot = std::make_shared<robot<Dim>>(Config);

    auto batchRobot = batchGASRobot<batchConfig>(Robot, streams, optixContext);
    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}
