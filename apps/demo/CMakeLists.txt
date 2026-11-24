# New-stack demos (ports of RTCD/Demo): rtcd_demo_discrete (old demoIAS / demoRAY) and rtcd_demo_curve (old
# demoLinContinuous / demoQuadContinuous), driven by setup JSON files (setups/demo_{ias,ray,linear,quad}.json: the
# old demo parameters, dense shelf). Public API only: they link `rtcd` and nothing else. Smoke-tested in tests/apps.
# Options: demo_common.h.
rtcd_new_stack_status()

if(RTCD_HAVE_CHECKER)
  add_library(rtcd_demo_common STATIC demo_common.cpp demo_common.h)
  target_link_libraries(rtcd_demo_common PUBLIC rtcd)
  set_target_properties(rtcd_demo_common PROPERTIES FOLDER "rtcd/apps")
  rtcd_register_new_target(rtcd_demo_common)
  rtcd_add_app(rtcd_demo_discrete SOURCES demo_discrete.cpp LIBS rtcd_demo_common)
  rtcd_add_app(rtcd_demo_curve SOURCES demo_curve.cpp LIBS rtcd_demo_common)
else()
  message(STATUS "rtcd_demo_discrete / rtcd_demo_curve: skipped (CollisionChecker not implemented in this tree)")
endif()
