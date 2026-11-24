# tiny_obj_loader.h is pinned to tinyobjloader commit 3bb554cf (2025-01-29, "Correct multi-material parsing",
# release branch, header version 2.0.0). Later revisions of the release branch no longer compile with MSVC 14.52
# (error C3615 in the embedded fast_float, min/max macro clash). Bump the commit and the hash together.
set(TINYOBJLOADER_COMMIT 3bb554cf74428d7db13418b4aca1b9752a1d2be8)
set(TINYOBJLOADER_SHA1 5C53EE20986800D83D92B2D876D8C6DCAA9BF416)

if(EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/tiny_obj_loader.h")
message(STATUS "tiny_obj_loader.h already exists")
else()
message(STATUS "Downloading tiny_obj_loader.h (tinyobjloader ${TINYOBJLOADER_COMMIT})")
file(DOWNLOAD
    https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/${TINYOBJLOADER_COMMIT}/tiny_obj_loader.h
    ${PROJECT_SOURCE_DIR}/3rdparty/tiny_obj_loader.h
    EXPECTED_HASH SHA1=${TINYOBJLOADER_SHA1}
    STATUS TINYOBJLOADER_STATUS)
list(GET TINYOBJLOADER_STATUS 0 TINYOBJLOADER_STATUS_CODE)
if(NOT TINYOBJLOADER_STATUS_CODE EQUAL 0)
    file(REMOVE ${PROJECT_SOURCE_DIR}/3rdparty/tiny_obj_loader.h)
    message(FATAL_ERROR "Downloading tiny_obj_loader.h failed: ${TINYOBJLOADER_STATUS}")
endif()
endif()

# The old stack adds 3rdparty/ as a directory include path (RTCD/CMakeLists.txt); new-tree targets link this
# interface target instead.
if(NOT TARGET rtcd_tinyobjloader)
  add_library(rtcd_tinyobjloader INTERFACE)
  target_include_directories(rtcd_tinyobjloader SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/3rdparty)
endif()
