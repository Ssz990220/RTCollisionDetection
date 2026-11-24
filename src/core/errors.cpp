// rtcd_core: throw helpers behind RTCD_CUDA_CHECK / RTCD_CU_CHECK / RTCD_OPTIX_CHECK(_LOG) (rtcd/errors.h),
// error names and sticky classification.
#include "core_internal.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table.h>
#include <optix_types.h>

#include <cstdio>
#include <string>

// The function table is defined in optix_function_table.cpp (its own object file, see there). Declared here so
// optixErrorText can check whether optixInit() has filled it.
extern "C" OptixFunctionTable OPTIX_FUNCTION_TABLE_SYMBOL;

namespace rtcd::detail {

    namespace {
        // Runtime and driver enumerators of the sticky errors share their numeric values; the table is written
        // with the runtime names and checked against the driver names here.
        static_assert(int(cudaErrorIllegalAddress) == int(CUDA_ERROR_ILLEGAL_ADDRESS));
        static_assert(int(cudaErrorLaunchFailure) == int(CUDA_ERROR_LAUNCH_FAILED));
        static_assert(int(cudaErrorLaunchTimeout) == int(CUDA_ERROR_LAUNCH_TIMEOUT));
        static_assert(int(cudaErrorAssert) == int(CUDA_ERROR_ASSERT));
        static_assert(int(cudaErrorHardwareStackError) == int(CUDA_ERROR_HARDWARE_STACK_ERROR));
        static_assert(int(cudaErrorIllegalInstruction) == int(CUDA_ERROR_ILLEGAL_INSTRUCTION));
        static_assert(int(cudaErrorMisalignedAddress) == int(CUDA_ERROR_MISALIGNED_ADDRESS));
        static_assert(int(cudaErrorInvalidAddressSpace) == int(CUDA_ERROR_INVALID_ADDRESS_SPACE));
        static_assert(int(cudaErrorInvalidPc) == int(CUDA_ERROR_INVALID_PC));
        static_assert(int(cudaErrorECCUncorrectable) == int(CUDA_ERROR_ECC_UNCORRECTABLE));

        std::string where(const char* expr, const char* file, int line) {
            std::string s = expr ? expr : "<call>";
            s += " failed at ";
            s += file ? file : "?";
            s += ":";
            s += std::to_string(line);
            return s;
        }
    } // namespace

    bool isStickyCudaCode(int code) noexcept {
        switch (code) {
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorAssert:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorECCUncorrectable:
            return true;
        default:
            return false;
        }
    }

    std::string cudaRuntimeErrorText(int code) {
        const auto e      = static_cast<cudaError_t>(code);
        const char* name  = cudaGetErrorName(e);
        const char* descr = cudaGetErrorString(e);
        std::string s     = name ? name : "unknown cudaError_t";
        s += " (";
        s += descr ? descr : "no description";
        s += ", code " + std::to_string(code) + ")";
        return s;
    }

    std::string cudaDriverErrorText(int code) {
        const char* name  = nullptr;
        const char* descr = nullptr;
        if (cuGetErrorName(static_cast<CUresult>(code), &name) != CUDA_SUCCESS) {
            name = nullptr;
        }
        if (cuGetErrorString(static_cast<CUresult>(code), &descr) != CUDA_SUCCESS) {
            descr = nullptr;
        }
        std::string s = name ? name : "unknown CUresult";
        s += " (";
        s += descr ? descr : "no description";
        s += ", code " + std::to_string(code) + ")";
        return s;
    }

    std::string optixErrorText(int code) {
        std::string s;
        // optixGetErrorName / optixGetErrorString go through the function table, which is empty before optixInit().
        if (OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorName && OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorString) {
            const char* name  = OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorName(static_cast<OptixResult>(code));
            const char* descr = OPTIX_FUNCTION_TABLE_SYMBOL.optixGetErrorString(static_cast<OptixResult>(code));
            s = name ? name : "unknown OptixResult";
            s += " (";
            s += descr ? descr : "no description";
            s += ", code " + std::to_string(code) + ")";
        } else {
            s = "OptixResult " + std::to_string(code) + " (OptiX not initialized, no name available)";
        }
        return s;
    }

    [[noreturn]] void throwCudaErrorMessage(int code, const std::string& message) {
        const bool sticky = isStickyCudaCode(code);
        if (!sticky) {
            // A failing runtime call also records the error as the "last error"; consume it so the next
            // RTCD_CUDA_CHECK(cudaGetLastError()) after a kernel launch does not report this error again.
            // (Sticky errors cannot be cleared; they are reported again by every later call.)
            (void) cudaGetLastError();
        }
        std::string what = message + ": " + cudaRuntimeErrorText(code);
        if (sticky) {
            what += " [sticky: the CUDA context is unusable; every object built on it is dead]";
        }
        throw CudaError(what, code, /*driver=*/false, sticky);
    }

    [[noreturn]] void throwCudaError(int code, const char* expr, const char* file, int line) {
        throwCudaErrorMessage(code, where(expr, file, line));
    }

    [[noreturn]] void throwCuError(int code, const char* expr, const char* file, int line) {
        const bool sticky = isStickyCudaCode(code);
        std::string what  = where(expr, file, line) + ": " + cudaDriverErrorText(code);
        if (sticky) {
            what += " [sticky: the CUDA context is unusable; every object built on it is dead]";
        }
        throw CudaError(what, code, /*driver=*/true, sticky);
    }

    [[noreturn]] void throwOptixError(int code, const char* expr, const char* file, int line, const char* log) {
        std::string what = where(expr, file, line) + ": " + optixErrorText(code);
        std::string logText;
        if (log && log[0] != '\0') {
            logText = log;
            what += "\nOptiX log:\n" + logText;
        }
        throw OptixError(what, code, std::move(logText));
    }

    void logReleaseError(const char* what, int code, bool driver) noexcept {
        try {
            // cudaErrorCudartUnloading / CUDA_ERROR_DEINITIALIZED: the process is exiting and CUDA is already torn
            // down; the resources are gone anyway, nothing worth reporting.
            if ((!driver && code == cudaErrorCudartUnloading) || (driver && code == CUDA_ERROR_DEINITIALIZED)) {
                return;
            }
            const std::string text = std::string("rtcd: ") + what + " failed during release: "
                                   + (driver ? cudaDriverErrorText(code) : cudaRuntimeErrorText(code));
            Logger{}.log(LogLevel::Error, text);
        } catch (...) {
            // never throw from release paths
        }
    }

} // namespace rtcd::detail
