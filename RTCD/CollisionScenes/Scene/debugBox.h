#pragma once
#include <CollisionScenes/Scene/sceneBuilder.h>

namespace RTCD {
    template <typename T>
    void addObj(T& s) {
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        // pose.translate(Eigen::Vector3f(0.85, 0, 0.4));
        std::unique_ptr<obstacle> shelf =
            std::make_unique<RTCD::obstacle>(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/debugBox.obj"), pose);
        s->addObstacle(std::move(shelf));

        s->uploadToDevice();
    }

} // namespace RTCD
