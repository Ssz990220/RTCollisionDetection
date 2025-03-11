#pragma once
#include <stddef.h>

namespace RTCD {
    struct batchSceneConfig {
        bool compactable = false;
        size_t NSTREAM   = 4; // Number of streams
        size_t BATCHSIZE = 128;

        constexpr batchSceneConfig() = default;
        constexpr batchSceneConfig(size_t NSTREAM) : compactable(false), NSTREAM(NSTREAM), BATCHSIZE(0) {}
        constexpr batchSceneConfig(size_t NSTREAM, size_t BATCHSIZE)
            : compactable(true), NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE) {}
        constexpr batchSceneConfig(bool compactable, size_t NSTREAM, size_t BATCHSIZE)
            : compactable(compactable), NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE) {}
    };
} // namespace RTCD
