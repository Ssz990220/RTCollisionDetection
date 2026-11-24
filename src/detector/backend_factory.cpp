// src/detector: createBackendForMode (backend_factory.h). Keep this the only function of this file (see header).
#include "detector/backend_factory.h"

#include <rtcd/errors.h>

#include <string>

namespace rtcd::detail {

    std::unique_ptr<Backend> createBackendForMode(const BackendCreateInfo& info) {
        if (info.config == nullptr) {
            throw Error(ErrorKind::Internal, "createBackendForMode: null config");
        }
        switch (info.config->mode) {
        case Mode::Mesh: return createMeshBackend(info);
        case Mode::Ray: return createRayBackend(info);
        case Mode::LinearCurve:
        case Mode::QuadraticCurve:
        case Mode::CubicCurve: return createCurveBackend(info);
        }
        throw Error(ErrorKind::Internal,
            "createBackendForMode: unknown mode " + std::to_string(static_cast<unsigned>(info.config->mode)));
    }

} // namespace rtcd::detail
