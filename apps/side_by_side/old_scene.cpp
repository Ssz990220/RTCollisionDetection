// OLD-STACK TU (deleted with the old stack): the compile-time scene of this executable, selected like the old
// benchmarks with SHELF / SHELFSIMPLE / DENSESHELF. The scene headers define non-inline functions, so they are
// included in this TU only.
#include "old_runner.h"

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
