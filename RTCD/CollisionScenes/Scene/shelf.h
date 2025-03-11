#pragma once
#include <CollisionScenes/Scene/sceneBuilder.h>

namespace RTCD {
    template <typename T>
    void addObj(T& s, bool loop) {
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        // pose.translate(Eigen::Vector3f(0.85, 0, 0.4));
        pose.translate(Eigen::Vector3f(0.45, 0, 0.4));
        std::unique_ptr<obstacle> shelf = std::make_unique<RTCD::obstacle>(
            CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/shelves.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(shelf));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(0, -0.6, 0));
        std::unique_ptr<obstacle> bin1 =
            std::make_unique<RTCD::obstacle>(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/bin.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(bin1));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(0, 0.6, 0));
        std::unique_ptr<obstacle> bin2 =
            std::make_unique<RTCD::obstacle>(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/bin.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(bin2));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(-0.4, 0, 0));
        std::unique_ptr<obstacle> Bob =
            std::make_unique<RTCD::obstacle>(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Bob.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(Bob));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(-0.4, 0, 0));
        std::unique_ptr<obstacle> Cow =
            std::make_unique<RTCD::obstacle>(CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Cow.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(Cow));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(-0.4, 0, 0));
        std::unique_ptr<obstacle> Fish = std::make_unique<RTCD::obstacle>(
            CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Fish.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(Fish));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(-0.4, 0, 0));
        std::unique_ptr<obstacle> Sheep = std::make_unique<RTCD::obstacle>(
            CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Sheep.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(Sheep));

        pose = Eigen::Affine3f::Identity();
        pose.translate(Eigen::Vector3f(-0.4, 0, 0));
        std::unique_ptr<obstacle> Board = std::make_unique<RTCD::obstacle>(
            CONCAT_PATHS(PROJECT_BASE_DIR, "/models/curobo/Snakeboard.obj"), pose, 1.0, loop);
        s->addObstacle(std::move(Board));

        s->uploadToDevice();
    }

} // namespace RTCD
