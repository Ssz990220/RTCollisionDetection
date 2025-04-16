
#pragma once
#include <array>
#include <robot/robotConfig.h>
#include "config.h"

inline constexpr uint8_t Dim = 7;
inline constexpr uint8_t nLinks = 8;
inline constexpr std::array<float, 12> baseT{ 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };

inline constexpr auto lowerBound = std::array<float, Dim>{ -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f };
inline constexpr auto upperBound = std::array<float, Dim>{ 2.9671f, 1.8326f, 2.9671f, 0.0f, 2.9671f, 3.8223f, 2.9671f };
inline constexpr auto jointTypes = std::array<uint8_t, Dim>{ 0, 0, 0, 0, 0, 0, 0 };
inline constexpr auto jointPreTfIdx = std::array<uint8_t, Dim>{ 0, 1, 2, 3, 4, 5, 6 };

inline constexpr std::array<float,Dim*12> sinMask{
    0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f
};

inline constexpr std::array<float,Dim*12> cosMask{
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f
};

inline constexpr std::array<float,Dim*12> oneMask{
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.333f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, -0.316f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0825f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, -0.0825f, 0.0f, 0.0f, 1.0f, 0.384f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.088f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

inline constexpr RTCD::robotConfig<Dim> Config{
    "panda",
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/modified_panda.urdf"),
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/merged_meshes"),
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/spheres"),
    baseT,
    sinMask,
    cosMask,
    oneMask
};
