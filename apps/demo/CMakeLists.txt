# Demos: rtcd_demo_discrete (poses, mesh / ray mode) and rtcd_demo_curve (trajectories, curve modes), driven by
# setup JSON files (configs/setups/demo_{ias,ray,linear,quad}.json: the old stack's demo parameters, dense shelf).
# Public API only: they link `rtcd` and nothing else. Smoke-tested in tests/apps. Options: demo_common.h.
add_library(rtcd_demo_common STATIC demo_common.cpp demo_common.h)
target_link_libraries(rtcd_demo_common PUBLIC rtcd)
set_target_properties(rtcd_demo_common PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_demo_common)
rtcd_add_app(rtcd_demo_discrete SOURCES demo_discrete.cpp LIBS rtcd_demo_common)
rtcd_add_app(rtcd_demo_curve SOURCES demo_curve.cpp LIBS rtcd_demo_common)
