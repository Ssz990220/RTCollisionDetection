# Side-by-side runner (Wave 1 e): old stack and new stack timed in the same process (sbs_main.cpp header).
# One executable per old compile-time scene: rtcd_sbs_shelf, rtcd_sbs_shelf_simple, rtcd_sbs_shelf_dense.
# Deleted (or reduced to the new half) together with the old stack.
rtcd_new_stack_status()

# Old-stack half, one library per old compile-time scene: the old headers define non-inline functions, so the
# profiles and the scene must share one TU (old_profiles.cpp includes old_scene.inl). rtcd_sbs_old_<scene>.
set(rtcd_sbs_defines SHELF SHELFSIMPLE DENSESHELF)
set(rtcd_sbs_names shelf shelf_simple shelf_dense)
foreach(define name IN ZIP_LISTS rtcd_sbs_defines rtcd_sbs_names)
  add_library(rtcd_sbs_old_${name} STATIC old_profiles.cpp old_runner.h old_scene.inl)
  rtcd_use_old_stack(rtcd_sbs_old_${name})
  target_compile_definitions(rtcd_sbs_old_${name} PRIVATE ${define})
  target_include_directories(rtcd_sbs_old_${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(rtcd_sbs_old_${name} CCCuda CCRay) # the old detector loads its OptiX IR from lib/ptx at run time
  set_target_properties(rtcd_sbs_old_${name} PROPERTIES FOLDER "rtcd/apps")
  rtcd_register_new_target(rtcd_sbs_old_${name})
endforeach()

# Profile -> new CheckerConfig mapping (plain structs; unit-tested in tests/apps).
add_library(rtcd_sbs_common STATIC profile_config.cpp profile_config.h old_runner.h)
target_link_libraries(rtcd_sbs_common PUBLIC rtcd_testsupport rtcd_config)
target_include_directories(rtcd_sbs_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(rtcd_sbs_common PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_sbs_common)

# Compile check of the new half with the CollisionChecker path enabled (always, part of AllNew; not linked).
add_library(rtcd_sbs_new_compile OBJECT new_runner.cpp)
target_link_libraries(rtcd_sbs_new_compile PRIVATE rtcd rtcd_testsupport)
target_compile_definitions(rtcd_sbs_new_compile PRIVATE RTCD_HAVE_CHECKER=1)
set_target_properties(rtcd_sbs_new_compile PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_sbs_new_compile)

if(RTCD_HAVE_BASE)
  foreach(name shelf shelf_simple shelf_dense)
    rtcd_add_app(rtcd_sbs_${name}
      SOURCES sbs_main.cpp new_runner.cpp
      LIBS rtcd_sbs_old_${name} rtcd_sbs_common rtcd_testsupport rtcd_json rtcd_robot rtcd_scene rtcd)
    rtcd_new_stack_definitions(rtcd_sbs_${name})
  endforeach()
else()
  message(STATUS "rtcd_sbs_*: not built (core / config / json / geometry / scene / robot not all implemented)")
endif()
