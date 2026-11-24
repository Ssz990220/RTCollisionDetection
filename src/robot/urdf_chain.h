#pragma once
// src/robot/urdf_chain.h -- PRIVATE to rtcd_robot. URDF -> serial kinematic chain -> KinematicsTable, without any
// geometry loading (so the table derivation is testable on its own; tests/robot uses it directly).
//
// Rules: include/rtcd/robot_model.h ("URDF -> kinematic chain"). Summary of what this file decides:
//   * The URDF is walked from its root. While the current root link has no geometry of the selected kind
//     (MeshSource) and its only child joint is fixed, that fixed joint is folded into `rootFixed` and the walk
//     moves on: rootFixed is "the fixed chain from the URDF root to geometry link 0" (identity for Panda).
//   * From a chain link L, every link reachable through fixed joints only is part of L: it is either merged
//     (it has geometry; its transform relative to L is recorded) or a pure frame (no geometry; dropped). The
//     actuated joints found on that fixed subtree must be at most one; its child starts chain link L + 1.
//   * Joint transform: A = (fixed transforms from L to the joint's parent link) * (joint origin), k = unit axis.
//     T_j(q) = A Rot(k, q) = cos(q) (-A K^2) + sin(q) (A K) + A (I + K^2)  (rotation parts; translation in the
//     "one" mask), all in double, |v| < 1e-9 -> +0.0 exactly, then rounded to float.
//   * Child order of a link is urdfdom's child_joints order (urdfdom builds it from a std::map keyed by joint
//     name, i.e. joint-name order, not document order). It matters only for the order of merged links.

#include <rtcd/geometry.h>
#include <rtcd/limits.h>
#include <rtcd/robot_model.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace urdf {
    class ModelInterface;
    class Link;
} // namespace urdf

namespace rtcd::robot_detail {

    // Row-major 3x4 rigid transform in double.
    using Affine = std::array<double, 12>;
    constexpr Affine kIdentityAffine{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    Affine compose(const Affine& a, const Affine& b); // a * b

    // A URDF link whose geometry belongs to a chain link, with its pose in the chain link's frame.
    struct GeometryPart {
        std::shared_ptr<const urdf::Link> link;
        Affine inChainLink = kIdentityAffine;
    };

    struct ChainLink {
        std::string name;                 // URDF name of the chain link (the link that starts it)
        std::vector<GeometryPart> parts;  // parts[0] is the chain link itself (always present, may lack geometry);
                                          // then merged links in walk order
        std::vector<std::string> mergedLinks; // names of parts[1..] that have geometry (= LinkGeometry::mergedLinks)
    };

    struct ChainJoint {
        std::string name;
        bool continuous = false;
        double lower = 0, upper = 0;   // radians (continuous: [-pi, pi])
        Affine origin = kIdentityAffine; // A: chain link j frame -> joint frame (before the rotation)
        std::array<double, 3> axis{0, 0, 1}; // unit axis in the joint frame
    };

    struct Chain {
        std::string robotName;
        Affine rootFixed = kIdentityAffine;
        std::vector<ChainLink> links;   // dof + 1
        std::vector<ChainJoint> joints; // dof
        std::shared_ptr<const urdf::ModelInterface> model; // keeps the urdf links alive
    };

    // Parses `urdfFile` (IoError if missing or not parseable) and builds the chain. `source` selects which URDF
    // element counts as "geometry" (decides merge vs. pure frame, and the root fixed chain).
    Chain loadChain(const std::filesystem::path& urdfFile, MeshSource source = MeshSource::Visual);

    // Builds the chain from an already parsed model (same rules). `where` names the file for messages.
    Chain buildChain(std::shared_ptr<const urdf::ModelInterface> model, MeshSource source, const std::string& where);

    // The table of a chain: masks per joint, base = basePose * rootFixed (double, snapped, rounded).
    KinematicsTable kinematicsFromChain(const Chain& chain, const Transform& basePose = Transform{});

    // Mask values of one joint before rounding (exposed for tests): cos, sin, one masks in double, snapped.
    struct JointMasks {
        Affine cosMask{}, sinMask{}, oneMask{};
    };
    JointMasks jointMasks(const Affine& origin, const std::array<double, 3>& unitAxis);

    inline constexpr double kSnapTolerance = 1e-9;

} // namespace rtcd::robot_detail
