# rtcd_regression: golden regression driven by configs/setups/regression_<mode>.json (regression_main.cpp header).
rtcd_add_app(rtcd_regression SOURCES regression_main.cpp engines.cpp
  LIBS rtcd rtcd_testsupport rtcd_json rtcd_robot rtcd_scene rtcd_backend_mesh rtcd_backend_ray rtcd_backend_curve)
