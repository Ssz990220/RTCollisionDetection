# rtcd_testsupport (Wave 1 e): include/rtcd/internal/reference_executor.h (ReferenceExecutor,
# ReferenceBatchRunner) plus test / benchmark utilities: cached inputs from data/motions exactly as
# RTCD/CollisionDetector/Test/ccRegression.cpp loads them, golden read / compare, flag-diff printing, legacy env
# knobs, timing statistics. Utility headers live here and are included as <support/...> (tests/ is on the PUBLIC
# include path). Added from the top-level CMakeLists.txt (also used by apps/), not by the tests/ folder scan.
#
# Object-file split matters: tests that only use the host utilities must not pull reference_executor.obj (which
# needs OptiX / rtcd_core at link time), so every utility is its own .cpp.
add_library(rtcd_testsupport STATIC
  paths.cpp
  inputs.cpp
  golden.cpp
  options.cpp
  timing.cpp
  reference.cpp
  reference_executor.cpp
)
target_include_directories(rtcd_testsupport
  PUBLIC  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_testsupport
  PUBLIC  rtcd_core rtcd_config rtcd_robot rtcd_scene)
set_target_properties(rtcd_testsupport PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_testsupport)

# ---------------------------------------------------------------------------------------------------------------
# Helpers for the apps (apps/regression, apps/bench, apps/side_by_side) and tests/apps, tests/testsupport. CMake
# functions are global, so the app folders (configured after this one) can call them.
# ---------------------------------------------------------------------------------------------------------------

# rtcd_module_implemented(<out> <module>): TRUE if src/<module> has a source other than placeholder.cpp/.cu,
# i.e. the module's implementation has landed in this tree. Re-globbed at build time (CONFIGURE_DEPENDS).
function(rtcd_module_implemented out module)
  file(GLOB _srcs CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/${module}/*.cpp" "${PROJECT_SOURCE_DIR}/src/${module}/*.cu")
  list(FILTER _srcs EXCLUDE REGEX "/placeholder\\.(cpp|cu)$")
  if(_srcs OR RTCD_ASSUME_NEW_STACK_IMPLEMENTED)
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()

set(RTCD_ASSUME_NEW_STACK_IMPLEMENTED OFF CACHE BOOL
  "Treat every new-stack module as implemented (link the apps even while src/<x> holds only a placeholder)")

# rtcd_new_stack_status(): sets in the caller's scope
#   RTCD_HAVE_<MODULE> for CORE CONFIG JSON GEOMETRY SCENE ROBOT DETECTOR BACKEND_MESH BACKEND_RAY BACKEND_CURVE,
#   RTCD_HAVE_BASE     core + config + json + geometry + scene + robot (everything below the backends),
#   RTCD_HAVE_CHECKER  base + detector + all three backends (rtcd_detector links all of them),
#   RTCD_HAVE_ANY_BACKEND.
function(rtcd_new_stack_status)
  set(base TRUE)
  foreach(m core config json geometry scene robot detector backend_mesh backend_ray backend_curve)
    rtcd_module_implemented(have ${m})
    string(TOUPPER ${m} M)
    set(RTCD_HAVE_${M} ${have} PARENT_SCOPE)
    set(have_${m} ${have})
  endforeach()
  foreach(m core config json geometry scene robot)
    if(NOT have_${m})
      set(base FALSE)
    endif()
  endforeach()
  set(checker ${base})
  foreach(m detector backend_mesh backend_ray backend_curve)
    if(NOT have_${m})
      set(checker FALSE)
    endif()
  endforeach()
  set(any FALSE)
  if(have_backend_mesh OR have_backend_ray OR have_backend_curve)
    set(any TRUE)
  endif()
  set(RTCD_HAVE_BASE ${base} PARENT_SCOPE)
  set(RTCD_HAVE_CHECKER ${checker} PARENT_SCOPE)
  set(RTCD_HAVE_ANY_BACKEND ${any} PARENT_SCOPE)
endfunction()

# rtcd_new_stack_definitions(<target>): PRIVATE compile definitions RTCD_HAVE_<MODULE>=0/1, RTCD_HAVE_CHECKER,
# so a TU can compile the code paths whose symbols will link.
function(rtcd_new_stack_definitions target)
  rtcd_new_stack_status()
  foreach(M BASE CHECKER DETECTOR BACKEND_MESH BACKEND_RAY BACKEND_CURVE)
    if(RTCD_HAVE_${M})
      target_compile_definitions(${target} PRIVATE RTCD_HAVE_${M}=1)
    else()
      target_compile_definitions(${target} PRIVATE RTCD_HAVE_${M}=0)
    endif()
  endforeach()
endfunction()

# rtcd_use_old_stack(<target>): lets a new-tree target include and link the old stack (frozen reference under
# RTCD/). The old stack's include paths are directory-scoped in RTCD/CMakeLists.txt and do not reach the new tree,
# so they are repeated here: RTCD/, 3rdparty/ (tiny_obj_loader.h) and the build dir (config.h). Links the old
# CollisionDetector interface target (meshes, utils, rtcd_legacy_kernels, OptiX, its OptiX-IR targets).
# USE_MULTI_THREADING is defined for every TU of the target (old regression / benchmarks were built with it and
# it changes the CollisionDetector layout, so all TUs of one binary must agree).
# Targets using this are deleted together with the old stack.
function(rtcd_use_old_stack target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/RTCD ${PROJECT_SOURCE_DIR}/3rdparty ${CMAKE_BINARY_DIR})
  target_link_libraries(${target} PRIVATE CollisionDetector)
  # NOMINMAX: optix_stubs.h pulls in windows.h, whose min / max macros break old headers included after it
  # (e.g. Planner/constant.h) depending on include order.
  target_compile_definitions(${target} PRIVATE USE_MULTI_THREADING NOMINMAX)
endfunction()
