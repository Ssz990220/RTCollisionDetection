# rtcd_robot (Wave 1 d): include/rtcd/robot_model.h (URDF via urdfdom -> chain -> KinematicsTable; link geometry
# through rtcd_geometry) and include/rtcd/internal/kinematics.h (FK / OBB / trajectory kernels, .cu files here).
add_library(rtcd_robot STATIC
  placeholder.cpp
  urdf_chain.h
  urdf_chain.cpp
)
target_include_directories(rtcd_robot
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_robot
  PUBLIC  rtcd_geometry
  PRIVATE urdfdom::urdf_parser urdfdom::urdfdom_model Eigen3::Eigen)
set_target_properties(rtcd_robot PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_robot)
