#include <CollisionScenes/Scene/denseShelf.h>
#include <Utils/Test/testUtils.h>

using namespace RTCD;
int main() {
    defaultInitOptix();
    CUcontext cudaContext;
    OptixDeviceContext optixContext;
    createDefaultContext(cudaContext, optixContext);

    std::unique_ptr<scene> s = buildScene();
    s->uploadToDevice();
    s->wrapScene(optixContext);
    return 0;
}
