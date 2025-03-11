#pragma once
#include <array>
#include <Robot/robotConfig.h>
#include "config.h"

inline constexpr int Dim = 7;
inline constexpr std::array<float, 12> baseT{ 1,0,0,0,0,1,0,0,0,0,1,0};

inline constexpr auto lowerBound = std::array<float, Dim>{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
inline constexpr auto upperBound = std::array<float, Dim>{2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973};

inline constexpr std::array<float,Dim*12> sinMask{
    0, -1, 0, 0,		1, 0, 0, 0,		0, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		-1, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		1, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		1, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		-1, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		1, 0, 0, 0,
    0, -1, 0, 0,		0, 0, 0, 0,		1, 0, 0, 0};

inline constexpr std::array<float,Dim*12> cosMask{
1, 0, 0, 0,		0, 1, 0, 0,		0, 0, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, -1, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, 1, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, 1, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, -1, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, 1, 0, 0,
1, 0, 0, 0,		0, 0, 0, 0,		0, 1, 0, 0};

inline constexpr std::array<float,Dim*12> oneMask{
    0, 0, 0, 0,			0, 0, 0, 0,			0, 0, 1, 0.333f,
    0, 0, 0, 0,			0, 0, 1, 0,			0, 0, 0, 0,
    0, 0, 0, 0,			0, 0, -1, -0.316f,	0, 0, 0, 0,
    0, 0, 0, 0.0825f,	0, 0, -1, 0,		0, 0, 0, 0,
    0, 0, 0, -0.0825f,	0, 0, 1, 0.384f,	0, 0, 0, 0,
    0, 0, 0, 0,			0, 0, -1, 0,		0, 0, 0, 0,
    0, 0, 0, 0.088f,	0, 0, -1, 0,		0, 0, 0, 0
};

inline constexpr RTCD::robotConfig<Dim> Config{
    "Panda",
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/panda.urdf"),
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/meshes/Simplified/watertight/"),
    CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/spheres/"),
    baseT,
    sinMask,
    cosMask,
    oneMask
};