if(EXISTS "${PROJECT_SOURCE_DIR}/3rdparty/tiny_obj_loader.h")
message(STATUS "tiny_obj_loader.h already exists")
else()
message(STATUS "Downloading tiny_obj_loader.h")
file(DOWNLOAD
    https://github.com/tinyobjloader/tinyobjloader/raw/release/tiny_obj_loader.h
    ${PROJECT_SOURCE_DIR}/3rdparty/tiny_obj_loader.h)
endif()

include_directories(${PROJECT_SOURCE_DIR}/3rdparty)