#pragma once
// src/detector (private): the default backend factory of CollisionChecker -- THE mode switch of the detector,
// called once per checker at construction.
//
// It lives alone in backend_factory.cpp, the only object file of rtcd_detector that references
// createMeshBackend / createRayBackend / createCurveBackend (implemented by rtcd_backend_mesh / _ray / _curve).
// The detector core (detector_core.cpp) takes its factory as a parameter and never references this function,
// so tests of the core with a fake backend link without the three backend libraries.

#include <rtcd/backend.h>

#include <memory>

namespace rtcd::detail {

    // Mesh -> createMeshBackend, Ray -> createRayBackend, Linear / Quadratic / CubicCurve -> createCurveBackend.
    std::unique_ptr<Backend> createBackendForMode(const BackendCreateInfo& info);

} // namespace rtcd::detail
