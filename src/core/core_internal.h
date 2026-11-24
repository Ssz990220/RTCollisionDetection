#pragma once
// src/core/core_internal.h -- private helpers of rtcd_core (not part of the contract; other modules must not
// include this header).

#include <rtcd/errors.h>
#include <rtcd/types.h>

#include <string>

namespace rtcd::detail {

    // True for CUDA errors that leave the context unusable (asynchronous device faults). The runtime (cudaError_t)
    // and driver (CUresult) enumerators of these errors have the same numeric values, so one table serves both.
    bool isStickyCudaCode(int code) noexcept;

    // "cudaErrorMemoryAllocation (out of memory)" for a runtime code, "CUDA_ERROR_... (...)" for a driver code.
    std::string cudaRuntimeErrorText(int code);
    std::string cudaDriverErrorText(int code);
    // "OPTIX_ERROR_... (...)"; works before optixInit() (falls back to the numeric code).
    std::string optixErrorText(int code);

    // Throws CudaError(code, runtime) with a caller-provided message (the text of the code is appended). Clears
    // the runtime's last-error state for non-sticky codes so a later cudaGetLastError() does not report it again.
    [[noreturn]] void throwCudaErrorMessage(int code, const std::string& message);

    // Logs through the default logger (stderr) at LogLevel::Error; used by destructors (never throws).
    void logReleaseError(const char* what, int code, bool driver) noexcept;

} // namespace rtcd::detail
