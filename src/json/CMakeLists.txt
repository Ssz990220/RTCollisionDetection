# rtcd_json (Wave 1 b): include/rtcd/json_config.h -- JSON (checker config, scene description, robot block,
# setup files, overrides, dumps) <-> the plain structs of include/rtcd. The only library that links nlohmann-json
# (PRIVATE: the public header does not include it). Paths in a file are resolved relative to that file.
add_library(rtcd_json STATIC
  json_config.cpp
)
target_include_directories(rtcd_json
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_json
  PUBLIC  rtcd_config
  PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(rtcd_json PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_json)
if(MSVC)
  target_compile_options(rtcd_json PRIVATE /W4 /WX /permissive- /utf-8)
endif()
