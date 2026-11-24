# Tests of rtcd_robot. The Panda results are compared with stored references written from the old stack
# (tests/data/reference; tests/data/README.md).

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
