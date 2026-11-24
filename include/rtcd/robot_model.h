#pragma once
// rtcd/robot_model.h -- the robot, loaded at runtime from its description (URDF via urdfdom) plus geometry
// directories. Replaces the hand-written RTCD/Robot/models/pandas.h (masks, limits, paths) and RTCD::robot<DOF>.
//
//   RobotDesc   where things are and in which format (what the JSON "robot" block holds).
//   RobotModel  the loaded robot: DOF, joint names and limits, the kinematics table the kernels consume, and per
//               link the mesh, the spheres and the OBB, all in host memory.
//
// RobotModel is HOST-ONLY and immutable: it owns no device memory and needs no Context, so it can be loaded and
// tested without a GPU and shared (shared_ptr<const RobotModel>) by any number of checkers on any Contexts.
// Each backend uploads what it needs (kinematics table into its own device buffer, link meshes into its GAS
// inputs, sphere arrays, OBB templates) at its construction.
//
// Threading: all const member functions are thread-safe.
//
// ---------------------------------------------------------------------------------------------------------------
// Loading rules (library rtcd_robot, src/robot, Wave 1 d; mesh / sphere loading and OBB fitting through the
// rtcd_geometry functions declared in rtcd/geometry.h)
// ---------------------------------------------------------------------------------------------------------------
// URDF -> kinematic chain:
//   * Joints: revolute and continuous are actuated (continuous limits = [-pi, pi]); fixed joints are folded into
//     the transforms; prismatic, floating and planar joints throw UnsupportedError naming the joint.
//   * Links attached to their parent by a fixed joint are MERGED into the parent link: their geometry (mesh
//     vertices, spheres) is transformed into the parent link frame and appended (mesh vertex / triangle indices
//     offset; edges recomputed on the merged mesh). Links without geometry attached by fixed joints are pure
//     frames and simply disappear.
//   * After merging, the tree must be a serial chain: link 0 = the URDF root, link j + 1 = child of actuated
//     joint j. More than one actuated branch throws UnsupportedError (old code threw on any branching).
//     DOF = number of actuated joints, 1 <= DOF <= kMaxDof (else UnsupportedError). numLinks = DOF + 1.
//   * Joint order (= order of joint values in a pose) is chain order root -> tip.
//   * Kinematics table: for joint j with parent-side fixed chain A (all fixed transforms from geometry link j to
//     joint j, including the joint origin) and unit axis k, T_j(q) = A * Rot(k, q). Because
//     Rot(k, q) = (I + K^2) + sin(q) K - cos(q) K^2 (K = cross-product matrix of k), T_j is linear in cos q and
//     sin q, giving cosMask[j] = -A K^2 (rotation part only), sinMask[j] = A K (rotation part only),
//     oneMask[j] = A (I + K^2) with A's translation. All computed in double, then every entry with
//     |v| < 1e-9 is set to exactly 0 and the rest rounded to float. For panda.urdf the table must equal the
//     hand-written masks of RTCD/Robot/models/pandas.h bit for bit (Wave 1 d test).
//   * base = RobotDesc::basePose * (fixed chain from the URDF root to geometry link 0, identity for Panda).
// Geometry:
//   * Mesh of a link: the URDF element selected by MeshSource (default visual) must be a <mesh>; the file is
//     RobotDesc::mesh.files[linkName] if present, else mesh.dir / <file name of the URDF mesh filename>
//     (old rule: the URDF's directory part is dropped, e.g. "meshes/visual/link7_nohand.obj" -> "link7_nohand.obj").
//     The element's <origin> and <mesh scale> are applied (identity for Panda; the old code ignored them).
//     Format "obj" only (tinyobjloader, triangulate). Every link of the chain (link 0 included) must have a mesh
//     in v1 (UnsupportedError otherwise).
//   * Spheres of a link (only if RobotDesc::spheres is set): file = spheres.files[linkName] if present, else
//     spheres.dir / (linkName + ".txt"). Format "xyzr": UTF-8 text, one sphere per line "x y z r" (four floats,
//     whitespace separated, link frame, meters); empty lines and lines whose first character is '#' are skipped;
//     any other line that does not parse as exactly four floats throws IoError(path, line). r > 0 required. A
//     missing sphere file throws IoError. A link may have zero spheres only if its file is empty.
//   * Link OBB: best-fitting OBB of the link mesh vertices, exactly the old computeBestFittingOBB
//     (RTCD/Meshes/Obb.h). Not collapsed here: CheckerConfig::collapseRootLinkObb is applied by the backends.

#include <rtcd/geometry.h>
#include <rtcd/limits.h>
#include <rtcd/types.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rtcd {

    // Where one kind of link geometry lives and in which format.
    struct GeometrySource {
        std::filesystem::path dir;                                 // resolved against the JSON file
        std::string format;                                        // "obj" for meshes, "xyzr" for spheres
        std::map<std::string, std::filesystem::path> files;        // optional per-link override: link name -> file
                                                                   // (relative entries resolved against dir)
    };

    enum class MeshSource : std::uint8_t { Visual, Collision }; // which URDF element names the link mesh

    struct RobotDesc {
        std::filesystem::path description;  // URDF file
        GeometrySource mesh{{}, "obj", {}};  // required
        MeshSource meshSource = MeshSource::Visual;
        std::optional<GeometrySource> spheres; // required by the curve modes, optional otherwise
        Transform basePose;                  // robot root -> world (old robotConfig::baseTransform, identity)
    };

    struct LinkGeometry {
        std::string name;                       // URDF name of the chain link
        std::vector<std::string> mergedLinks;   // URDF links merged into it by fixed joints (may be empty)
        TriangleMesh mesh;                      // link frame
        std::vector<Sphere> spheres;            // link frame, file order (merged links appended in URDF order:
                                                // the order of urdfdom's child-joint container, i.e. sorted by
                                                // joint name, not file order)
        Obb obb;                                // best-fit OBB of mesh.vertices, link frame
        std::filesystem::path meshFile;
        std::filesystem::path sphereFile;       // empty if no spheres were requested
    };

    class RobotModel {
    public:
        // Loads and validates (see the header comment). Throws IoError, UnsupportedError, InvalidArgument.
        static std::shared_ptr<const RobotModel> load(const RobotDesc& desc, const Logger& logger = {});

        const RobotDesc& desc() const noexcept { return desc_; }
        const std::string& name() const noexcept { return name_; } // URDF <robot name>
        std::uint32_t dof() const noexcept { return table_.dof; }
        std::uint32_t numLinks() const noexcept { return table_.numLinks; }

        std::span<const std::string> jointNames() const noexcept { return jointNames_; }
        std::span<const float> lowerLimits() const noexcept { return lower_; } // radians, URDF <limit>
        std::span<const float> upperLimits() const noexcept { return upper_; }

        const KinematicsTable& kinematics() const noexcept { return table_; }

        std::span<const LinkGeometry> links() const noexcept { return links_; } // numLinks() entries, chain order

        bool hasSpheres() const noexcept { return hasSpheres_; }
        // Flat sphere arrays over all links, link-major (link 0's spheres first), file order within a link. This is
        // the old robot<DOF> sphereCentersTemplate / sphereRadiusTemplate / sphereIdxToLinkIdx order.
        std::uint32_t totalSpheres() const noexcept { return static_cast<std::uint32_t>(sphereLink_.size()); }
        std::span<const float> sphereCenters() const noexcept { return sphereCenters_; } // 3 floats per sphere
        std::span<const float> sphereRadii() const noexcept { return sphereRadii_; }
        std::span<const std::int32_t> sphereLink() const noexcept { return sphereLink_; } // sphere -> link index

        // Host reference forward kinematics with the same formula as the kernels (float, std::sin / std::cos, so
        // not bit-identical to __sincosf): q has dof() floats, out receives numLinks() * 12 floats (link poses in
        // the world, row-major 3x4). For tests and tools. Throws InvalidArgument on size mismatch.
        void linkPoses(std::span<const float> q, std::span<float> out) const;

    private:
        RobotModel() = default;
        RobotDesc desc_;
        std::string name_;
        std::vector<std::string> jointNames_;
        std::vector<float> lower_, upper_;
        KinematicsTable table_{};
        std::vector<LinkGeometry> links_;
        bool hasSpheres_ = false;
        std::vector<float> sphereCenters_;
        std::vector<float> sphereRadii_;
        std::vector<std::int32_t> sphereLink_;
    };

} // namespace rtcd
