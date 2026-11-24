# rtcd_config (Wave 1 b): CheckerConfig::validate() / resolved() of include/rtcd/config.h. Plain C++, no
# dependencies besides the public headers (no CUDA, no JSON).
add_library(rtcd_config STATIC
  config.cpp
)
target_include_directories(rtcd_config
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_target_properties(rtcd_config PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_config)
if(MSVC)
  target_compile_options(rtcd_config PRIVATE /W4 /WX /permissive- /utf-8)
endif()
