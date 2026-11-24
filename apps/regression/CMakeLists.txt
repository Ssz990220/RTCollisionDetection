# rtcd_regression (Wave 1 e): new-stack golden regression driven by configs/setups/regression_<mode>.json
# (regression_main.cpp header). Engines compile in only when what they call is implemented in this tree
# (rtcd_new_stack_definitions); rtcd_regression_compile checks the full code against the declarations always.
rtcd_new_stack_status()

add_library(rtcd_regression_compile OBJECT regression_main.cpp engines.cpp)
target_link_libraries(rtcd_regression_compile PRIVATE rtcd rtcd_testsupport)
target_compile_definitions(rtcd_regression_compile PRIVATE
  RTCD_HAVE_CHECKER=1 RTCD_HAVE_BACKEND_MESH=1 RTCD_HAVE_BACKEND_RAY=1 RTCD_HAVE_BACKEND_CURVE=1)
set_target_properties(rtcd_regression_compile PROPERTIES FOLDER "rtcd/apps")
rtcd_register_new_target(rtcd_regression_compile)

if(RTCD_HAVE_BASE)
  set(libs rtcd_testsupport rtcd_json rtcd_robot rtcd_scene)
  if(RTCD_HAVE_CHECKER)
    list(APPEND libs rtcd)
  endif()
  foreach(b mesh ray curve)
    string(TOUPPER ${b} B)
    if(RTCD_HAVE_BACKEND_${B})
      list(APPEND libs rtcd_backend_${b})
    endif()
  endforeach()
  rtcd_add_app(rtcd_regression SOURCES regression_main.cpp engines.cpp LIBS ${libs})
  rtcd_new_stack_definitions(rtcd_regression)
endif()
