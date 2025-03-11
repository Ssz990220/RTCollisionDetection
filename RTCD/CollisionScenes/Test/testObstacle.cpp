#include "config.h"
#include "optix_function_table_definition.h"
#include <CollisionScenes/obstacle.h>

int main() {
    RTCD::cube cube(0.1f, 0.1f, 0.1f);
    cube.uploadToDevice();
    RTCD::obstacle obstacle(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/PipeLong.obj"));
    obstacle.uploadToDevice();
}
