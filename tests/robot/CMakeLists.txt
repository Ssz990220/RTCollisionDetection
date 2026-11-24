# Tests of rtcd_robot (Wave 1 d owns this folder).
#
# Tests marked OLD-STACK PARITY include old-stack headers (RTCD/...) and / or link rtcd_legacy_kernels as the
# reference implementation; they are deleted (or reduced to the new-stack half) together with the old stack.

# Old-stack include paths needed by the parity tests: RTCD/ (Robot/models/pandas.h, Robot/robot.cuh, ...) and the
# build dir (config.h, included by pandas.h).
set(rtcd_robot_test_legacy_includes ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})

# URDF -> chain -> KinematicsTable; pandas.h bit-for-bit parity (no GPU).
rtcd_add_new_test(test_robot_kinematics_table LABEL new_unit
  SOURCES test_kinematics_table.cpp
  LIBS rtcd_robot urdfdom::urdf_parser urdfdom::urdfdom_model)
target_include_directories(test_robot_kinematics_table PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${rtcd_robot_test_legacy_includes})

# RobotModel::load: Panda vs the old per-link meshModel loader; loading rules on synthetic URDFs (no GPU).
rtcd_add_new_test(test_robot_model LABEL new_unit
  SOURCES test_robot_model.cpp
  LIBS rtcd_robot meshes)
target_include_directories(test_robot_model PRIVATE ${rtcd_robot_test_legacy_includes})

# New FK kernels vs the old robot.cu kernels on the cached regression poses / KNN trajectories (GPU).
rtcd_add_new_test(test_robot_kinematics_kernels LABEL new_gpu
  SOURCES test_kinematics_kernels.cpp  LIBS rtcd_robot rtcd_legacy_kernels)
target_include_directories(test_robot_kinematics_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${rtcd_robot_test_legacy_includes})
