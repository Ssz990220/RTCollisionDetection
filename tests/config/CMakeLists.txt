# Tests of rtcd_config (Wave 1 b owns this folder): CheckerConfig defaults, every validate() rule, resolved() per
# mode. Host only, no GPU, no JSON.
rtcd_add_new_test(test_config_unit LABEL new_unit
  SOURCES test_config.cpp
  LIBS rtcd_config)
if(MSVC)
  target_compile_options(test_config_unit PRIVATE /W4 /WX /permissive- /utf-8)
endif()
