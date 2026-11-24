// URDF -> serial chain -> KinematicsTable (rules: urdf_chain.h, include/rtcd/robot_model.h).
#include "robot/urdf_chain.h"

#include <rtcd/errors.h>

#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <utility>

namespace rtcd::robot_detail {

    namespace {

        bool hasGeometry(const urdf::Link& link, MeshSource source) {
            if (source == MeshSource::Visual) {
                return link.visual && link.visual->geometry;
            }
            return link.collision && link.collision->geometry;
        }

        Affine poseToAffine(const urdf::Pose& p) {
            double x = p.rotation.x, y = p.rotation.y, z = p.rotation.z, w = p.rotation.w;
            const double n = std::sqrt(x * x + y * y + z * z + w * w);
            if (n > 0) {
                x /= n, y /= n, z /= n, w /= n;
            } else {
                x = y = z = 0, w = 1;
            }
            return Affine{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p.position.x,
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p.position.y,
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p.position.z};
        }

        const char* jointTypeName(int type) {
            switch (type) {
            case urdf::Joint::REVOLUTE: return "revolute";
            case urdf::Joint::CONTINUOUS: return "continuous";
            case urdf::Joint::PRISMATIC: return "prismatic";
            case urdf::Joint::FLOATING: return "floating";
            case urdf::Joint::PLANAR: return "planar";
            case urdf::Joint::FIXED: return "fixed";
            default: return "unknown";
            }
        }

        double snap(double v) { return std::fabs(v) < kSnapTolerance ? 0.0 : v; }

        std::shared_ptr<const urdf::Link> childOf(const urdf::ModelInterface& model, const urdf::Joint& j,
            const std::string& where) {
            auto c = model.getLink(j.child_link_name);
            if (!c) {
                throw IoError("URDF joint '" + j.name + "' names a missing child link '" + j.child_link_name + "'", where);
            }
            return c;
        }

    } // namespace

    Affine compose(const Affine& a, const Affine& b) {
        Affine r{};
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 4; ++c) {
                double v = a[i * 4 + 0] * b[0 + c] + a[i * 4 + 1] * b[4 + c] + a[i * 4 + 2] * b[8 + c];
                if (c == 3) {
                    v += a[i * 4 + 3];
                }
                r[i * 4 + c] = v;
            }
        }
        return r;
    }

    JointMasks jointMasks(const Affine& A, const std::array<double, 3>& k) {
        // K = [k]x, K2 = K K. Rot(k, q) = (I + K2) + sin(q) K - cos(q) K2 (Rodrigues).
        const double K[3][3]  = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}};
        double K2[3][3]       = {};
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 3; ++c) {
                K2[i][c] = K[i][0] * K[0][c] + K[i][1] * K[1][c] + K[i][2] * K[2][c];
            }
        }
        JointMasks m;
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 3; ++c) {
                double rk = 0, rk2 = 0;
                for (int t = 0; t < 3; ++t) {
                    rk += A[i * 4 + t] * K[t][c];
                    rk2 += A[i * 4 + t] * K2[t][c];
                }
                m.cosMask[i * 4 + c] = snap(-rk2);
                m.sinMask[i * 4 + c] = snap(rk);
                m.oneMask[i * 4 + c] = snap(A[i * 4 + c] + rk2);
            }
            m.cosMask[i * 4 + 3] = 0.0;
            m.sinMask[i * 4 + 3] = 0.0;
            m.oneMask[i * 4 + 3] = snap(A[i * 4 + 3]);
        }
        return m;
    }

    Chain buildChain(std::shared_ptr<const urdf::ModelInterface> model, MeshSource source, const std::string& where) {
        if (!model) {
            throw IoError("URDF model is empty", where);
        }
        Chain chain;
        chain.robotName = model->getName();
        auto root       = model->getRoot();
        if (!root) {
            throw IoError("URDF has no root link", where);
        }

        // Fixed chain from the URDF root to geometry link 0 (pure frames above the first link with geometry).
        std::shared_ptr<const urdf::Link> L = root;
        while (!hasGeometry(*L, source) && L->child_joints.size() == 1 && L->child_joints[0]->type == urdf::Joint::FIXED) {
            const auto& j   = *L->child_joints[0];
            chain.rootFixed = compose(chain.rootFixed, poseToAffine(j.parent_to_joint_origin_transform));
            L               = childOf(*model, j, where);
        }

        for (;;) {
            ChainLink cl;
            cl.name = L->name;
            cl.parts.push_back({L, kIdentityAffine});

            std::shared_ptr<const urdf::Joint> actuated;
            Affine actuatedParent = kIdentityAffine;

            // Depth-first walk of the fixed subtree of L (pre-order, urdfdom child order).
            std::vector<std::pair<std::shared_ptr<const urdf::Link>, Affine>> stack{{L, kIdentityAffine}};
            while (!stack.empty()) {
                auto [M, T] = stack.back();
                stack.pop_back();
                // Push children in reverse so they are visited in order.
                std::vector<std::pair<std::shared_ptr<const urdf::Link>, Affine>> fixedChildren;
                for (const auto& jp : M->child_joints) {
                    const urdf::Joint& j = *jp;
                    if (j.mimic) {
                        throw UnsupportedError(where + ": URDF joint '" + j.name + "' is a mimic joint (not supported)");
                    }
                    switch (j.type) {
                    case urdf::Joint::FIXED: {
                        auto C        = childOf(*model, j, where);
                        const Affine TC = compose(T, poseToAffine(j.parent_to_joint_origin_transform));
                        fixedChildren.emplace_back(C, TC);
                        break;
                    }
                    case urdf::Joint::REVOLUTE:
                    case urdf::Joint::CONTINUOUS:
                        if (actuated) {
                            throw UnsupportedError(where + ": the kinematic tree branches at link '" + cl.name
                                                   + "' (actuated joints '" + actuated->name + "' and '" + j.name
                                                   + "'); only serial chains are supported");
                        }
                        actuated       = jp;
                        actuatedParent = T;
                        break;
                    default:
                        throw UnsupportedError(where + ": URDF joint '" + j.name + "' has type "
                                               + jointTypeName(j.type)
                                               + " (only revolute, continuous and fixed joints are supported)");
                    }
                }
                for (auto it = fixedChildren.rbegin(); it != fixedChildren.rend(); ++it) {
                    stack.push_back(*it);
                }
                if (M != L) {
                    if (hasGeometry(*M, source)) {
                        cl.parts.push_back({M, T});
                        cl.mergedLinks.push_back(M->name);
                    }
                }
            }
            chain.links.push_back(std::move(cl));
            if (!actuated) {
                break;
            }

            ChainJoint cj;
            cj.name       = actuated->name;
            cj.continuous = actuated->type == urdf::Joint::CONTINUOUS;
            if (cj.continuous) {
                cj.lower = -std::numbers::pi;
                cj.upper = std::numbers::pi;
            } else {
                if (!actuated->limits) {
                    throw IoError("URDF revolute joint '" + actuated->name + "' has no <limit>", where);
                }
                cj.lower = actuated->limits->lower;
                cj.upper = actuated->limits->upper;
            }
            cj.origin = compose(actuatedParent, poseToAffine(actuated->parent_to_joint_origin_transform));
            const double ax = actuated->axis.x, ay = actuated->axis.y, az = actuated->axis.z;
            const double n  = std::sqrt(ax * ax + ay * ay + az * az);
            if (!(n > 0)) {
                throw InvalidArgument(where + ": URDF joint '" + actuated->name + "' has a zero axis");
            }
            cj.axis = {ax / n, ay / n, az / n};
            chain.joints.push_back(cj);
            if (chain.joints.size() > kMaxDof) {
                throw UnsupportedError(where + ": the robot has more than kMaxDof = " + std::to_string(kMaxDof)
                                       + " actuated joints");
            }
            L = childOf(*model, *actuated, where);
        }

        if (chain.joints.empty()) {
            throw UnsupportedError(where + ": the robot has no actuated (revolute / continuous) joint");
        }
        chain.model = std::move(model);
        return chain;
    }

    Chain loadChain(const std::filesystem::path& urdfFile, MeshSource source) {
        std::ifstream in(urdfFile, std::ios::binary);
        if (!in) {
            throw IoError("cannot open URDF file", urdfFile);
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        std::shared_ptr<const urdf::ModelInterface> model = urdf::parseURDF(ss.str());
        if (!model) {
            throw IoError("cannot parse URDF file (urdfdom rejected it)", urdfFile);
        }
        return buildChain(std::move(model), source, urdfFile.string());
    }

    KinematicsTable kinematicsFromChain(const Chain& chain, const Transform& basePose) {
        KinematicsTable t{};
        t.dof      = static_cast<std::uint32_t>(chain.joints.size());
        t.numLinks = t.dof + 1;
        for (std::size_t j = 0; j < chain.joints.size(); ++j) {
            const JointMasks m = jointMasks(chain.joints[j].origin, chain.joints[j].axis);
            for (std::size_t e = 0; e < kTransformFloats; ++e) {
                t.cosMask[j][e] = static_cast<float>(m.cosMask[e]);
                t.sinMask[j][e] = static_cast<float>(m.sinMask[e]);
                t.oneMask[j][e] = static_cast<float>(m.oneMask[e]);
            }
        }
        Affine base{};
        for (std::size_t e = 0; e < kTransformFloats; ++e) {
            base[e] = static_cast<double>(basePose.m[e]);
        }
        base = compose(base, chain.rootFixed);
        for (std::size_t e = 0; e < kTransformFloats; ++e) {
            t.base[e] = static_cast<float>(snap(base[e]));
        }
        return t;
    }

} // namespace rtcd::robot_detail
