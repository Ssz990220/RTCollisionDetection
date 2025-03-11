#pragma once
#include "config.h"
#include <CollisionScenes/scene.h>
#include <memory>

namespace RTCD {

    template <typename T>
    void addObj(T& s, bool loop);

    std::unique_ptr<scene> buildScene(bool loop = false) {
        std::unique_ptr<scene> s = std::make_unique<scene>();
        addObj(s, loop);
        return s;
    }

    std::unique_ptr<scene> buildUniScene(bool loop = false) {
        std::unique_ptr<scene> s = std::make_unique<scene>();
        addObj(s, loop);
        return s;
    }

    std::shared_ptr<scene> buildSharedScene(bool loop = false) {
        std::shared_ptr<scene> s = std::make_shared<scene>();
        addObj(s, loop);
        return s;
    }
} // namespace RTCD
