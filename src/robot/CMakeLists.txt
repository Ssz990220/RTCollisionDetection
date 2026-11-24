# rtcd_robot: robot model loaded at runtime (URDF via urdfdom), per-robot device kinematics tables, kinematics
# kernels (fused fkine / sphere / trajectory kernels).
add_library(rtcd_robot STATIC
  placeholder.cpp
)
target_include_directories(rtcd_robot
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_robot
  PUBLIC  rtcd_geometry
  PRIVATE urdfdom::urdf_parser urdfdom::urdfdom_model Eigen3::Eigen)
set_target_properties(rtcd_robot PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_robot)
