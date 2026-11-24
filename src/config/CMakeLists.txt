# rtcd_config: JSON (checker config, scene description) -> plain config structs of include/rtcd. Depends on the
# public headers and nlohmann-json only (the core library never sees JSON). Paths in a config are resolved relative
# to the config file.
add_library(rtcd_config STATIC
  placeholder.cpp
)
target_include_directories(rtcd_config
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rtcd_config
  PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(rtcd_config PROPERTIES FOLDER "rtcd")
rtcd_register_new_target(rtcd_config)
