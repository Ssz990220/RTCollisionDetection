// Included by old_profiles.cpp only (deleted with the old stack): the compile-time scene, selected like the old
// benchmarks with SHELF / SHELFSIMPLE / DENSESHELF. The old headers define non-inline functions (meshModel,
// obstacle, scene, tinyobjloader), so the whole old stack must live in ONE translation unit per binary; hence
// one rtcd_sbs_old_<scene> library per scene.

#if defined(DENSESHELF)
#include <CollisionScenes/Scene/denseShelf.h>
#define RTCD_SBS_SCENE_NAME "shelf_dense"
#elif defined(SHELFSIMPLE)
#include <CollisionScenes/Scene/shelfSimple.h>
#define RTCD_SBS_SCENE_NAME "shelf_simple"
#elif defined(SHELF)
#include <CollisionScenes/Scene/shelf.h>
#define RTCD_SBS_SCENE_NAME "shelf"
#else
#error "define one of SHELF, SHELFSIMPLE, DENSESHELF"
#endif

std::shared_ptr<RTCD::scene> rtcdSbsBuildOldScene(bool loop) { return RTCD::buildSharedScene(loop); }

const char* rtcd::sbs::oldSceneName() { return RTCD_SBS_SCENE_NAME; }
