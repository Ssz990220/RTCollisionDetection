#define TIME_IT
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "config.h"
#include <Meshes/mesh.h>
#include <iostream>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

int main(int argc, char* argv[]) {
    RTCD::meshModel model;
    if (argc < 2) {
        model.loadOBJ(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Snakeboard.obj"));
    } else {
        model.loadOBJ(std::string(PROJECT_BASE_DIR) + argv[1]);
    }
}
