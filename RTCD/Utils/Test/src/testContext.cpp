// #include <CollisionScenes/Scene/shelf.h>
#include <Utils/Test/testUtils.h>

inline constexpr size_t NSTREAM = 4;

int main() {
    auto [cudaContext, optixContext, streams] = createContextStream<NSTREAM>();
    return 0;
}
