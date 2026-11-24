// RobotModel::load and members (include/rtcd/robot_model.h). The chain / table part is in urdf_chain.cpp; this file
// adds the link geometry through the rtcd_geometry functions (loadObjMesh, computeEdges, fitObb, loadSpheres).
#include "robot/urdf_chain.h"

#include <rtcd/errors.h>
#include <rtcd/robot_model.h>

#include <urdf_model/model.h>

#include <cmath>
#include <string>
#include <utility>

namespace rtcd {

    namespace {

        using robot_detail::Affine;

        bool isIdentity(const Affine& a) { return a == robot_detail::kIdentityAffine; }

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

        Transform toTransform(const Affine& a) {
            Transform t;
            for (std::size_t e = 0; e < kTransformFloats; ++e) {
                t.m[e] = static_cast<float>(std::fabs(a[e]) < robot_detail::kSnapTolerance ? 0.0 : a[e]);
            }
            return t;
        }

        // The URDF element (visual / collision) of `link` selected by `source`; null if absent.
        const urdf::Visual* visualOf(const urdf::Link& l) { return l.visual.get(); }
        const urdf::Collision* collisionOf(const urdf::Link& l) { return l.collision.get(); }

        std::filesystem::path resolveFile(const GeometrySource& src, const std::string& linkName,
            const std::filesystem::path& defaultName) {
            if (auto it = src.files.find(linkName); it != src.files.end()) {
                return it->second.is_absolute() ? it->second : src.dir / it->second;
            }
            return src.dir / defaultName;
        }

        // Old rule: drop the URDF's directory part ("meshes/visual/link7_nohand.obj" -> "link7_nohand.obj"; also
        // strips "package://pkg/...").
        std::string meshBaseName(const std::string& urdfFilename) {
            const std::size_t pos = urdfFilename.find_last_of("/\\");
            return pos == std::string::npos ? urdfFilename : urdfFilename.substr(pos + 1);
        }

        void appendMesh(TriangleMesh& dst, const TriangleMesh& src) {
            const auto offset = static_cast<std::uint32_t>(dst.vertices.size());
            dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
            for (auto t : src.triangles) {
                dst.triangles.push_back({t[0] + offset, t[1] + offset, t[2] + offset});
            }
        }

    } // namespace

    std::shared_ptr<const RobotModel> RobotModel::load(const RobotDesc& desc, const Logger& logger) {
        if (desc.mesh.format != "obj") {
            throw UnsupportedError("robot: mesh format '" + desc.mesh.format + "' is not supported (only \"obj\")");
        }
        if (desc.spheres && desc.spheres->format != "xyzr") {
            throw UnsupportedError("robot: sphere format '" + desc.spheres->format + "' is not supported (only \"xyzr\")");
        }
        const std::string where     = desc.description.string();
        const robot_detail::Chain c = robot_detail::loadChain(desc.description, desc.meshSource);

        std::shared_ptr<RobotModel> m(new RobotModel());
        m->desc_  = desc;
        m->name_  = c.robotName;
        m->table_ = robot_detail::kinematicsFromChain(c, desc.basePose);
        for (const auto& j : c.joints) {
            m->jointNames_.push_back(j.name);
            m->lower_.push_back(static_cast<float>(j.lower));
            m->upper_.push_back(static_cast<float>(j.upper));
        }
        m->hasSpheres_ = desc.spheres.has_value();

        for (std::size_t li = 0; li < c.links.size(); ++li) {
            const robot_detail::ChainLink& cl = c.links[li];
            LinkGeometry g;
            g.name        = cl.name;
            g.mergedLinks = cl.mergedLinks;
            int meshParts = 0;
            for (const robot_detail::GeometryPart& part : cl.parts) {
                const urdf::Link& ul = *part.link;
                const urdf::Geometry* geom = nullptr;
                urdf::Pose origin;
                if (desc.meshSource == MeshSource::Visual) {
                    if (const urdf::Visual* v = visualOf(ul)) {
                        geom   = v->geometry.get();
                        origin = v->origin;
                    }
                    if (ul.visual_array.size() > 1) {
                        throw UnsupportedError(where + ": link '" + ul.name + "' has several <visual> elements");
                    }
                } else {
                    if (const urdf::Collision* v = collisionOf(ul)) {
                        geom   = v->geometry.get();
                        origin = v->origin;
                    }
                    if (ul.collision_array.size() > 1) {
                        throw UnsupportedError(where + ": link '" + ul.name + "' has several <collision> elements");
                    }
                }
                if (geom == nullptr) {
                    continue; // only parts[0] can lack geometry (pure frames are not parts)
                }
                if (geom->type != urdf::Geometry::MESH) {
                    throw UnsupportedError(where + ": the geometry of link '" + ul.name
                                           + "' is not a <mesh> (primitive shapes are not supported)");
                }
                const auto* mesh = static_cast<const urdf::Mesh*>(geom);
                const double sx = mesh->scale.x, sy = mesh->scale.y, sz = mesh->scale.z;
                if (!(sx > 0) || sx != sy || sx != sz) {
                    throw UnsupportedError(where + ": link '" + ul.name
                                           + "' has a non-uniform or non-positive mesh scale (only uniform scales)");
                }
                const std::filesystem::path file = resolveFile(desc.mesh, ul.name, meshBaseName(mesh->filename));
                const Affine poseA = robot_detail::compose(part.inChainLink, poseToAffine(origin));
                // Identity pose and unit scale: vertices exactly as read (old robot-link path; keeps -0.0f
                // coordinates that an affine product would turn into +0.0f).
                TriangleMesh tm = (isIdentity(poseA) && sx == 1.0)
                                    ? loadObjMesh(file)
                                    : loadObjMesh(file, toTransform(poseA), static_cast<float>(sx));
                if (meshParts == 0) {
                    g.mesh     = std::move(tm);
                    g.meshFile = file;
                } else {
                    appendMesh(g.mesh, tm);
                }
                ++meshParts;

                if (desc.spheres) {
                    const std::filesystem::path sf = resolveFile(*desc.spheres, ul.name, ul.name + ".txt");
                    std::vector<Sphere> s          = loadSpheres(sf, desc.spheres->format);
                    if (!isIdentity(part.inChainLink)) {
                        const Transform T = toTransform(part.inChainLink);
                        for (Sphere& sp : s) {
                            sp.center = T.apply(sp.center);
                        }
                    }
                    if (g.sphereFile.empty()) {
                        g.sphereFile = sf;
                    }
                    g.spheres.insert(g.spheres.end(), s.begin(), s.end());
                }
            }
            if (meshParts > 1) {
                computeEdges(g.mesh); // edges of the merged mesh
            }
            if (g.mesh.empty()) {
                throw UnsupportedError(where + ": chain link '" + cl.name + "' has no mesh (every link needs one)");
            }
            g.obb = fitObb(g.mesh.vertices);
            for (const Sphere& sp : g.spheres) {
                m->sphereCenters_.insert(m->sphereCenters_.end(), sp.center.begin(), sp.center.end());
                m->sphereRadii_.push_back(sp.radius);
                m->sphereLink_.push_back(static_cast<std::int32_t>(li));
            }
            m->links_.push_back(std::move(g));
        }

        logger.log(LogLevel::Info, "robot '" + m->name_ + "' loaded from " + where + ": dof " + std::to_string(m->dof())
                                       + ", links " + std::to_string(m->numLinks()) + ", spheres "
                                       + std::to_string(m->totalSpheres()));
        return m;
    }

    void RobotModel::linkPoses(std::span<const float> q, std::span<float> out) const {
        if (q.size() != table_.dof || out.size() != std::size_t(table_.numLinks) * kTransformFloats) {
            throw InvalidArgument("RobotModel::linkPoses: q needs " + std::to_string(table_.dof) + " floats (got "
                                  + std::to_string(q.size()) + "), out needs "
                                  + std::to_string(std::size_t(table_.numLinks) * kTransformFloats) + " (got "
                                  + std::to_string(out.size()) + ")");
        }
        robot_detail::hostLinkPoses(table_, q.data(), out.data());
    }

} // namespace rtcd
