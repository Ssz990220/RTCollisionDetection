#pragma once
#include <array>
#include <string_view>
namespace RTCD {

    // Robot Configuration:
    // DOF: Degrees of Freedom
    // Current implementation only support URDF with fixed base.
    // 1. All links should have meshes, and all links should have no more than one child link.
    //    This means, for a robot of N links, there should be N-1 joints.
    // 2. All links are assumed to be rotational. We will add support for prismatics and tree structures support in the
    //    future.
    // 3. The link meshes should be aligned with the link coordinate.
    template <int DOF>
    struct robotConfig {
        std::string_view name;
        std::string_view urdfPath;
        std::string_view meshDir;
        std::string_view sphereDir;
        std::array<float, 12> baseTransform;
        std::array<float, 12 * DOF> sinMask;
        std::array<float, 12 * DOF> cosMask;
        std::array<float, 12 * DOF> oneMask;
        // a transform that will never cause collision. Some where under ground.
        std::array<float, 12> safeTransform = {0.001, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.001, -0.1};

        constexpr robotConfig(std::string_view name, std::string_view urdf, std::string_view mesh,
            std::string_view sphereDir, const std::array<float, 12>& baseT, const std::array<float, 12 * DOF>& sin,
            const std::array<float, 12 * DOF>& cos, const std::array<float, 12 * DOF>& one)
            : name(name), urdfPath(urdf), meshDir(mesh), sphereDir(sphereDir), baseTransform(baseT), sinMask(sin),
              cosMask(cos), oneMask(one) {
            static_assert(DOF > 0, "DOF should be greater than 0");
            static_assert(DOF <= 8, "DOF should be no greater than 8, if so, please modify the size of cosMask, "
                                    "sinMask, oneMask in robot.cu");
        };

        int getDof() const { return DOF; }
    };
} // namespace RTCD
