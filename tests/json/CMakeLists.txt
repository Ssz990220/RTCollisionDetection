# Tests of rtcd_json (Wave 1 b owns this folder): the configs/ files, schema errors, path resolution, setups,
# overrides, dump round trips. Host only, no GPU. nlohmann-json is linked only to inspect dumps.
rtcd_add_new_test(test_json_unit LABEL new_unit
  SOURCES test_json.cpp
  LIBS rtcd_json nlohmann_json::nlohmann_json)
target_compile_definitions(test_json_unit PRIVATE RTCD_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
if(MSVC)
  target_compile_options(test_json_unit PRIVATE /W4 /WX /permissive- /utf-8)
endif()
