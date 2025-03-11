#pragma once
#undef near
#undef far

#include <Meshes/mesh.h>
#include <Robot/meshUtils.h>
#include <Robot/robot.cuh>
#include <Robot/robotConfig.h>
// #include "sweptVolume.cuh"
#include <array>
#include <numeric>
#include <string_view>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <vector>

#define SKIP_BASE

namespace RTCD {


    template <size_t DOF>
    class robot {
    public:
        std::string_view name;

    private: // Members
        bool sphereUploaded = false;
        bool meshUploaded   = false;
        bool OBBUploaded    = false;
        urdf::ModelInterfaceSharedPtr model;
        std::vector<urdf::JointConstSharedPtr> activeJoints;
        std::map<urdf::LinkConstSharedPtr, std::shared_ptr<meshModel>> linkMap;
        std::vector<std::shared_ptr<meshModel>> orderedLinks;
        std::map<urdf::JointSharedPtr, size_t> JointIdToIdx;
        const Eigen::Affine3d baseTransform = Eigen::Affine3d::Identity();
        size_t NumLinks                     = 0;
        size_t NumJoints                    = 0;
        size_t NumVisableLinks              = 0;
        size_t NumActiveJoints              = 0;

        // Bounding Box
        std::vector<OBB> linkOBBs;
        std::vector<OBB> linkSphereOBBs;
        CUDABuffer linkOBBsBuffer;


        // Sphere Representation
        size_t robotTotalSphereCount = 0;
        std::vector<int> linkSphereCount; // Number of spheres per link
        std::vector<float3> sphereCentersTemplate; // Template Sphere Center on CPU
        CUDABuffer sphereCentersTemplateBuffer; // Template Sphere Center on GPU
        std::vector<float> sphereRadiusTemplate; // Template Sphere Radius on CPU
        std::vector<int> sphereIdxToLinkIdx; // Map sphere primitive index to link index ON CPU
        CUDABuffer sphereIdxToLinkIdxBuffer; // Map sphere primitive index to link index on GPU

    public:
        robot() = default;
        robot(robotConfig<DOF> config);

        void uploadSphereArrays();
        void uploadOBBs();
        void uploadSphereOBBs();

        // Public Getters
        inline std::vector<std::shared_ptr<meshModel>>& getOrderedLinks() { return orderedLinks; }

        inline size_t getRobotTotalSphereCount() const { return robotTotalSphereCount; }

        inline CUdeviceptr getSphereCenters() const { return sphereCentersTemplateBuffer.d_pointer(); }

        inline std::vector<float>& getSphereRadii() { return sphereRadiusTemplate; }

        inline CUdeviceptr getSphereIdxMap() const { return sphereIdxToLinkIdxBuffer.d_pointer(); }

        inline CUdeviceptr getOBBs() const { return linkOBBsBuffer.d_pointer(); }

    private:
        void loadURDF(
            const std::string_view urdfPath, const std::string_view meshDir, const std::string_view sphereDir);
        void traverseInitRobot(
            urdf::LinkConstSharedPtr link, const std::string_view meshDir, const std::string_view sphereDir);
    };


}; // namespace RTCD

// Implementations
namespace RTCD {

    template <size_t DOF>
    robot<DOF>::robot(robotConfig<DOF> config) {
        name = config.name;
        loadURDF(config.urdfPath, config.meshDir, config.sphereDir);
        CUDA_SYNC_CHECK();
        setFkineMasks(config.cosMask.data(), config.sinMask.data(), config.oneMask.data(), config.cosMask.size());
        setBaseAffine(config.baseTransform);
        setSafeTransform(config.safeTransform);
    };


    template <size_t DOF>
    void robot<DOF>::loadURDF(
        const std::string_view urdfPath, const std::string_view meshDir, const std::string_view sphereDir) {
        const std::filesystem::path p(urdfPath);
        if (!std::filesystem::exists(p)) {
            std::cout << "Mesh file " << std::filesystem::absolute(p) << " does not exist!" << std::endl;
            exit(1);
        }
        model = urdf::parseURDFFile(std::string(urdfPath));
        traverseInitRobot(model->getRoot(), meshDir, sphereDir);

#ifdef SKIP_BASE
        linkOBBs[0].halfSize = make_float3(0.0f, 0.0f, 0.0f);
#endif
        sphereIdxToLinkIdx.resize(robotTotalSphereCount);
        int counter = 0;
        for (size_t i = 0; i < NumLinks; ++i) {
            for (size_t j = 0; j < linkSphereCount[i]; ++j) {
                sphereIdxToLinkIdx[counter] = i;
                counter++;
            }
        }
    };

    template <size_t DOF>
    void robot<DOF>::uploadSphereArrays() {
        if (sphereUploaded) {
            return;
        }
        sphereCentersTemplateBuffer.alloc_and_upload(sphereCentersTemplate);
        sphereIdxToLinkIdxBuffer.alloc_and_upload(sphereIdxToLinkIdx);
        sphereUploaded = true;
    };

    template <size_t DOF>
    void robot<DOF>::traverseInitRobot(
        urdf::LinkConstSharedPtr urdfLink, const std::string_view meshDir, const std::string_view sphereDir) {
        ++NumLinks;
        if (urdfLink->visual && urdfLink->visual->geometry->type == urdf::Geometry::MESH) {
            // link mesh representation
            std::shared_ptr<meshModel> link = std::make_shared<meshModel>(urdfLink, meshDir, sphereDir);
            ++NumVisableLinks;
            linkMap[urdfLink] = link;
            orderedLinks.push_back(link);

            // link sphere representation
            linkSphereCount.push_back(link->sphereCount);
            linkOBBs.push_back(link->obb);
            linkSphereOBBs.push_back(link->sphereOBB);
            robotTotalSphereCount += link->sphereCount;
            sphereCentersTemplate.insert(
                sphereCentersTemplate.end(), link->sphereOrigins.begin(), link->sphereOrigins.end());
            sphereRadiusTemplate.insert(sphereRadiusTemplate.end(), link->sphereRadii.begin(), link->sphereRadii.end());
        }
        if (urdfLink->parent_joint) { // If not Root Link
            ++NumJoints;
            if (urdfLink->parent_joint->type == urdf::Joint::REVOLUTE) {
                JointIdToIdx[urdfLink->parent_joint] = NumActiveJoints;
                activeJoints.push_back(urdfLink->parent_joint);
                ++NumActiveJoints;
            }
        }

        if (urdfLink->child_links.empty()) {
            return;
        }
        if (urdfLink->child_links.size() > 1) {
            throw("Robot has more than one child link"); // TODO: Add support for branching
        } else {
            for (const urdf::LinkSharedPtr& child : urdfLink->child_links) {
                this->traverseInitRobot(child, meshDir, sphereDir);
            }
        }
    };

    template <size_t DOF>
    void robot<DOF>::uploadOBBs() {
        if (OBBUploaded) {
            return;
        }
        linkOBBsBuffer.alloc_and_upload(linkOBBs);
        OBBUploaded = true;
    };

    template <size_t DOF>
    void robot<DOF>::uploadSphereOBBs() {
        if (OBBUploaded) {
            return;
        }
        linkOBBsBuffer.alloc_and_upload(linkSphereOBBs);
        OBBUploaded = true;
    };


} // namespace RTCD
