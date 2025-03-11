#include "optix_function_table.h"
#include "optix_function_table_definition.h"
#include <Meshes/mesh.h>
#include <Utils/Test/testUtils.h>
#include <config.h>
#include <optix_function_table.h>
using namespace RTCD;

int main() {
    auto m = meshModel(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/RA830/merged_meshes/wrist_3_merged_visual.obj"));

    OBB obb               = m.getOBB();
    std::vector<OBB> obbs = {obb};
    writeBin(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/linkOBB3.bin"), obbs);
}
