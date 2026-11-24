# Tests of rtcd_robot (Wave 1 d owns this folder).
#
# The Panda results are compared with stored references (tests/data/reference, written from the old stack; see
# tests/data/README.md). test_robot_kinematics_kernels_vs_old is the OLD-STACK parity test that wrote them; it is
# deleted together with the old stack.

# URDF -> chain -> KinematicsTable; the stored pandas.h table bit for bit (no GPU).
rtcd_add_new_test(test_robot_kinematics_table LABEL new_unit
  SOURCES test_kinematics_table.cpp
  LIBS rtcd_robot rtcd_testsupport urdfdom::urdf_parser urdfdom::urdfdom_model)
target_include_directories(test_robot_kinematics_table PRIVATE ${PROJECT_SOURCE_DIR}/src)

# RobotModel::load: Panda vs the stored link summaries (old per-link meshModel loader); loading rules on synthetic
# URDFs (no GPU).
rtcd_add_new_test(test_robot_model LABEL new_unit
  SOURCES test_robot_model.cpp
  LIBS rtcd_robot rtcd_testsupport)

# FK kernels on the cached regression poses / KNN trajectories vs the stored old-kernel outputs (GPU).
rtcd_add_new_test(test_robot_kinematics_kernels LABEL new_gpu
  SOURCES test_kinematics_kernels.cpp fk_reference_cases.h
  LIBS rtcd_robot rtcd_testsupport)
target_include_directories(test_robot_kinematics_kernels PRIVATE ${PROJECT_SOURCE_DIR}/src)

# OLD-STACK PARITY + REFERENCE GENERATOR (delete with the old stack): new FK kernels vs the old robot.cu kernels,
# derived table vs pandas.h; with RTCD_WRITE_REFERENCE=<dir> it writes panda_kinematics.txt and kinematics_fk.*.
# Old-stack include paths: RTCD/ (Robot/models/pandas.h, Robot/robot.cuh) and the build dir (config.h).
rtcd_add_new_test(test_robot_kinematics_kernels_vs_old LABEL new_gpu
  SOURCES test_kinematics_kernels_vs_old.cpp fk_reference_cases.h
  LIBS rtcd_robot rtcd_testsupport rtcd_legacy_kernels)
target_include_directories(test_robot_kinematics_kernels_vs_old PRIVATE
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
