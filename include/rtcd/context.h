#pragma once
// rtcd/context.h -- the GPU context: one CUDA device (its primary context), one OptiX device context, a pool of
// CUDA streams, and the Logger every object built on the Context uses.
//
// Ownership / lifetime:
//   * A Context is created with Context::create() and always held by std::shared_ptr. Every object that owns
//     device memory or OptiX objects (Scene, CollisionChecker, backends) keeps a shared_ptr<Context>, so the
//     Context outlives them.
//   * The Context does not own the CUDA primary context exclusively: it retains it (cuDevicePrimaryCtxRetain
//     semantics via the runtime) and releases its retain on destruction. Several Contexts on the same device are
//     allowed (each has its own OptiX device context and streams).
//   * Streams are leased: acquireStreams(n) returns a StreamLease of n streams used exclusively by the lessee
//     (one CollisionChecker); on destruction of the lease they return to the pool (after a stream sync).
//     Pool streams are created with cudaStreamCreate (default flags, i.e. they synchronize with the legacy
//     default stream; same as the old stack).
//
// Threading: Context methods are thread-safe (the pool is mutex protected). The handles it returns follow CUDA /
// OptiX rules (an OptiX device context may be used from several host threads).
//
// Errors: create() throws CudaError / OptixError (no device, driver too old, OptiX init failure).

#include <rtcd/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtcd {

    struct ContextOptions {
        int device = 0; // CUDA device ordinal
        // OptiX device-context options (old OptixCompileConfig::contextValidation / contextLogLevel).
        bool optixValidation = false; // OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
        int optixLogLevel    = 0;     // 0 = no OptiX log callback, 1..4 = optixDeviceContextSetLogCallback level;
                                      // OptiX messages are forwarded to `logger` (fatal/error -> Error,
                                      // warning -> Warning, print -> Info)
        Logger logger;                // used by everything built on this Context

        // Former compile-time defaults: Debug builds of the *including* TU get validation on.
        static ContextOptions buildDefault() {
            ContextOptions o;
#if defined(_DEBUG) || defined(DEBUG)
            o.optixValidation = true;
#endif
            return o;
        }
    };

    class Context;

    // Exclusive lease of `size()` streams of a Context's pool. Move-only.
    class StreamLease {
    public:
        StreamLease() noexcept = default;
        ~StreamLease(); // synchronizes the streams, returns them to the pool; never throws
        StreamLease(StreamLease&&) noexcept;
        StreamLease& operator=(StreamLease&&) noexcept;
        StreamLease(const StreamLease&)            = delete;
        StreamLease& operator=(const StreamLease&) = delete;

        std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(streams_.size()); }
        StreamHandle operator[](std::uint32_t i) const { return streams_.at(i); }
        const std::vector<StreamHandle>& streams() const noexcept { return streams_; }

    private:
        friend class Context;
        std::shared_ptr<Context> owner_;
        std::vector<StreamHandle> streams_;
    };

    struct DeviceInfo {
        std::string name;
        int ordinal              = 0;
        int computeMajor         = 0;
        int computeMinor         = 0;
        int multiprocessors      = 0;
        std::size_t globalMemory = 0; // bytes
    };

    class Context : public std::enable_shared_from_this<Context> {
    public:
        // Initializes CUDA on options.device (cudaSetDevice + cudaFree(0)), optixInit(), creates the OptiX device
        // context. Synchronous.
        static std::shared_ptr<Context> create(const ContextOptions& options = ContextOptions::buildDefault());

        ~Context(); // destroys the OptiX device context and pool streams; never throws
        Context(const Context&)            = delete;
        Context& operator=(const Context&) = delete;

        int device() const noexcept;
        const DeviceInfo& deviceInfo() const noexcept;
        CudaContextHandle cudaContext() const noexcept;
        OptixContextHandle optixContext() const noexcept;
        const Logger& logger() const noexcept;

        // Makes this Context's device current on the calling thread (cudaSetDevice). Every rtcd entry point that
        // touches CUDA calls this first, so user threads need not.
        void makeCurrent() const;

        // Leases n >= 1 streams (creates streams if the pool has fewer free ones).
        StreamLease acquireStreams(std::uint32_t n);

    private:
        Context() = default;
        friend class StreamLease;
        void returnStreams(std::vector<StreamHandle>& streams) noexcept;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rtcd
