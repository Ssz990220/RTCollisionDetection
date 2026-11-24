#pragma once
// rtcd/errors.h -- exception types and the error-checking policy of the new stack.
//
// POLICY (all modules, all build types, Release included):
//   1. Library code never calls exit(), abort() or std::terminate() and never prints-and-continues on failure:
//      every failure is reported by throwing one of the exceptions below (all derive from rtcd::Error, which
//      derives from std::runtime_error).
//   2. Every CUDA runtime / driver call and every OptiX call is checked, in every build type, with
//      RTCD_CUDA_CHECK / RTCD_CU_CHECK / RTCD_OPTIX_CHECK. The check is one host branch per call; it is not a
//      synchronization. Kernel launches are followed by RTCD_CUDA_CHECK(cudaGetLastError()) (launch-config errors).
//      No device-wide synchronization is added for error checking.
//   3. Asynchronous device faults surface at the next checked call on that stream / context (usually the final
//      stream synchronization of a check). They are reported as CudaError with sticky() == true; after a sticky
//      error the CUDA context is unusable: every object built on that Context is dead and the process should
//      treat the GPU as lost (CUDA cannot recover a context in-process).
//   4. Destructors never throw. Errors while releasing resources are reported through the Context's Logger at
//      LogLevel::Error and otherwise ignored (objects may be destroyed after CUDA teardown at process exit).
//   5. Precondition violations by the caller (null spans, too many items, wrong sizes) throw InvalidArgument or
//      CapacityError BEFORE any GPU work is enqueued, so the object stays usable.
//   6. Messages are complete sentences naming the object, the field / file and the offending value, e.g.
//      "checker config: curve.control_points = 20 must be < curve.trajectory_points = 16".
//
// The throw helpers are implemented in src/core (they look up CUDA / OptiX error names), so this header needs
// no CUDA or OptiX include.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace rtcd {

    enum class ErrorKind : std::uint8_t {
        Config,          // invalid configuration value or combination (ConfigError)
        Io,              // file missing / unreadable / malformed content (IoError)
        Capacity,        // a per-call value exceeds a capacity given at construction (CapacityError)
        InvalidArgument, // malformed call argument (InvalidArgument)
        Unsupported,     // valid input that the library does not support (UnsupportedError), e.g. prismatic joints
        Cuda,            // CUDA runtime / driver failure (CudaError)
        Optix,           // OptiX failure (OptixError)
        Internal,        // broken internal invariant (a bug in rtcd)
    };

    class Error : public std::runtime_error {
    public:
        Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
        ErrorKind kind() const noexcept { return kind_; }

    private:
        ErrorKind kind_;
    };

    // Invalid configuration. `field` is the dotted JSON path of the (first) offending field, e.g.
    // "ray.launch_width_cap", or "" when the problem is a combination. what() lists every problem found, one per
    // line (validation collects all problems before throwing).
    class ConfigError : public Error {
    public:
        ConfigError(const std::string& message, std::string field = {})
            : Error(ErrorKind::Config, message), field_(std::move(field)) {}
        const std::string& field() const noexcept { return field_; }

    private:
        std::string field_;
    };

    // File-level failure: missing / unreadable file, or content that cannot be parsed (JSON syntax, OBJ, URDF,
    // sphere or weight file). line == 0 when not applicable. what() = "<path>[:<line>[:<column>]]: <message>".
    class IoError : public Error {
    public:
        IoError(const std::string& message, std::filesystem::path path, std::size_t line = 0, std::size_t column = 0)
            : Error(ErrorKind::Io, message), path_(std::move(path)), line_(line), column_(column) {}
        const std::filesystem::path& path() const noexcept { return path_; }
        std::size_t line() const noexcept { return line_; }
        std::size_t column() const noexcept { return column_; }

    private:
        std::filesystem::path path_;
        std::size_t line_;
        std::size_t column_;
    };

    // A per-call value exceeds a capacity fixed at construction (e.g. 9000 items > max_items_per_check = 8192).
    class CapacityError : public Error {
    public:
        CapacityError(const std::string& message, std::string capacityName, std::size_t requested, std::size_t capacity)
            : Error(ErrorKind::Capacity, message), name_(std::move(capacityName)), requested_(requested),
              capacity_(capacity) {}
        const std::string& capacityName() const noexcept { return name_; }
        std::size_t requested() const noexcept { return requested_; }
        std::size_t capacity() const noexcept { return capacity_; }

    private:
        std::string name_;
        std::size_t requested_;
        std::size_t capacity_;
    };

    class InvalidArgument : public Error {
    public:
        explicit InvalidArgument(const std::string& message) : Error(ErrorKind::InvalidArgument, message) {}
    };

    class UnsupportedError : public Error {
    public:
        explicit UnsupportedError(const std::string& message) : Error(ErrorKind::Unsupported, message) {}
    };

    // CUDA failure. code() is the cudaError_t (runtime) or CUresult (driver) value; driver() tells which.
    // sticky() is true for errors that corrupt the context (cudaErrorIllegalAddress, cudaErrorLaunchFailure, ...,
    // as classified by src/core).
    class CudaError : public Error {
    public:
        CudaError(const std::string& message, int code, bool driver, bool sticky)
            : Error(ErrorKind::Cuda, message), code_(code), driver_(driver), sticky_(sticky) {}
        int code() const noexcept { return code_; }
        bool driver() const noexcept { return driver_; }
        bool sticky() const noexcept { return sticky_; }

    private:
        int code_;
        bool driver_;
        bool sticky_;
    };

    // OptiX failure. code() is the OptixResult; log() is the module / program-group / pipeline compile log if
    // the failing call produced one (it is also appended to what()).
    class OptixError : public Error {
    public:
        OptixError(const std::string& message, int code, std::string log = {})
            : Error(ErrorKind::Optix, message), code_(code), log_(std::move(log)) {}
        int code() const noexcept { return code_; }
        const std::string& log() const noexcept { return log_; }

    private:
        int code_;
        std::string log_;
    };

    namespace detail {
        // Implemented in src/core. `expr` is the stringified call, file / line its location.
        [[noreturn]] void throwCudaError(int code, const char* expr, const char* file, int line);
        [[noreturn]] void throwCuError(int code, const char* expr, const char* file, int line);
        [[noreturn]] void throwOptixError(int code, const char* expr, const char* file, int line, const char* log);
    } // namespace detail

} // namespace rtcd

// Checking macros. `call` is evaluated exactly once. Success is the value 0 for cudaError_t (cudaSuccess),
// CUresult (CUDA_SUCCESS) and OptixResult (OPTIX_SUCCESS). The local variable name is reserved
// (rtcd_check_result_) so the macros nest and never shadow caller variables (old OPTIX_CHECK declared `res`).
#define RTCD_CUDA_CHECK(call)                                                                                    \
    do {                                                                                                         \
        const int rtcd_check_result_ = static_cast<int>(call);                                                   \
        if (rtcd_check_result_ != 0) {                                                                           \
            ::rtcd::detail::throwCudaError(rtcd_check_result_, #call, __FILE__, __LINE__);                       \
        }                                                                                                        \
    } while (0)

#define RTCD_CU_CHECK(call)                                                                                      \
    do {                                                                                                         \
        const int rtcd_check_result_ = static_cast<int>(call);                                                   \
        if (rtcd_check_result_ != 0) {                                                                           \
            ::rtcd::detail::throwCuError(rtcd_check_result_, #call, __FILE__, __LINE__);                         \
        }                                                                                                        \
    } while (0)

#define RTCD_OPTIX_CHECK(call)                                                                                   \
    do {                                                                                                         \
        const int rtcd_check_result_ = static_cast<int>(call);                                                   \
        if (rtcd_check_result_ != 0) {                                                                           \
            ::rtcd::detail::throwOptixError(rtcd_check_result_, #call, __FILE__, __LINE__, nullptr);             \
        }                                                                                                        \
    } while (0)

// For optixModuleCreate / optixProgramGroupCreate / optixPipelineCreate: `log` is the char buffer passed to the
// call (NUL-terminated by OptiX); it is attached to the exception.
#define RTCD_OPTIX_CHECK_LOG(call, log)                                                                          \
    do {                                                                                                         \
        const int rtcd_check_result_ = static_cast<int>(call);                                                   \
        if (rtcd_check_result_ != 0) {                                                                           \
            ::rtcd::detail::throwOptixError(rtcd_check_result_, #call, __FILE__, __LINE__, (log));               \
        }                                                                                                        \
    } while (0)
