#pragma once
#include <Meshes/meshTypes.h>

namespace RTCD {

    typedef unsigned int uint;

    enum class curveEndPointType {
        Phantom,
        Repeated,
        Normal,
    };

    struct batchRobotConfig {
        size_t NSTREAM   = 4;
        size_t BATCHSIZE = 64;
        size_t DOF       = 6;
        BuildType BUILD  = BuildType::FAST_BUILD;
        LinkType TYPE    = LinkType::CUBIC_CURVE;

        // In case needed for GAS
        size_t curveDegree             = 3;
        size_t nTrajPts                = 0;
        size_t nCtrlPts                = 0;
        bool useOBB                    = true;
        curveEndPointType endPointType = curveEndPointType::Normal;

        constexpr batchRobotConfig(size_t NSTREAM, size_t BATCHSIZE, size_t DOF, BuildType BUILD, LinkType TYPE)
            : NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE), DOF(DOF), BUILD(BUILD), TYPE(TYPE){};

        constexpr batchRobotConfig(size_t NSTREAM, size_t BATCHSIZE, size_t DOF, BuildType BUILD, LinkType TYPE,
            size_t nTrajPts, size_t nCtrlPts)
            : NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE), DOF(DOF), BUILD(BUILD), TYPE(TYPE), nTrajPts(nTrajPts),
              nCtrlPts(nCtrlPts){};

        constexpr batchRobotConfig(
            size_t NSTREAM, size_t BATCHSIZE, size_t DOF, BuildType BUILD, LinkType TYPE, bool useOBB)
            : NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE), DOF(DOF), BUILD(BUILD), TYPE(TYPE), useOBB(useOBB){};

        constexpr batchRobotConfig(size_t NSTREAM, size_t BATCHSIZE, size_t DOF, BuildType BUILD, LinkType TYPE,
            size_t nTrajPts, size_t nCtrlPts, bool useOBB)
            : NSTREAM(NSTREAM), BATCHSIZE(BATCHSIZE), DOF(DOF), BUILD(BUILD), TYPE(TYPE), nTrajPts(nTrajPts),
              nCtrlPts(nCtrlPts), useOBB(useOBB){};
    };
} // namespace RTCD
