// rtcd_core: Context, StreamLease (rtcd/context.h). Replaces createContextStream<NSTREAM>() / defaultInitOptix() /
// context_log_cb of RTCD/Utils/Test/cudaCtxUtils.h.
#include <rtcd/context.h>
#include <rtcd/errors.h>

#include "core_internal.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <mutex>
#include <string>
#include <utility>

namespace rtcd {

    namespace {
        // optixInit() loads the OptiX library and fills the process-wide function table; do it once per process
        // (concurrent Context::create calls are safe).
        int initOptixOnce() {
            static std::once_flag flag;
            static int result = OPTIX_SUCCESS;
            std::call_once(flag, [] { result = static_cast<int>(optixInit()); });
            return result;
        }
    } // namespace

    struct Context::Impl {
        ContextOptions options;
        DeviceInfo info;
        CUdevice cuDevice      = 0;
        CUcontext cuContext    = nullptr; // retained primary context
        bool primaryRetained   = false;
        bool deviceSelected    = false; // cudaSetDevice(info.ordinal) succeeded in create()
        OptixDeviceContext optix = nullptr;

        std::mutex poolMutex;
        std::vector<StreamHandle> allStreams;  // every stream created by this Context (destroyed with it)
        std::vector<StreamHandle> freeStreams; // not leased
    };

    // ---- Context -----------------------------------------------------------------------------------------------

    std::shared_ptr<Context> Context::create(const ContextOptions& options) {
        if (options.device < 0) {
            throw InvalidArgument("Context::create: device ordinal " + std::to_string(options.device) + " is negative");
        }
        if (options.optixLogLevel < 0 || options.optixLogLevel > 4) {
            throw InvalidArgument("Context::create: optixLogLevel = " + std::to_string(options.optixLogLevel)
                                  + " must be in [0, 4]");
        }

        std::shared_ptr<Context> ctx(new Context());
        ctx->impl_          = std::make_unique<Impl>();
        Impl& impl          = *ctx->impl_;
        impl.options        = options;
        impl.info.ordinal   = options.device;

        // CUDA: select the device and initialize its primary context (old defaultInitOptix: cudaFree(0)).
        RTCD_CUDA_CHECK(cudaSetDevice(options.device));
        impl.deviceSelected = true;
        RTCD_CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp props{};
        RTCD_CUDA_CHECK(cudaGetDeviceProperties(&props, options.device));
        impl.info.name            = props.name;
        impl.info.computeMajor    = props.major;
        impl.info.computeMinor    = props.minor;
        impl.info.multiprocessors = props.multiProcessorCount;
        impl.info.globalMemory    = props.totalGlobalMem;

        // Own retain of the primary context (released in ~Context), so the CUcontext handle stays valid for the
        // lifetime of this Context whatever else the process does with the runtime.
        RTCD_CU_CHECK(cuDeviceGet(&impl.cuDevice, options.device));
        RTCD_CU_CHECK(cuDevicePrimaryCtxRetain(&impl.cuContext, impl.cuDevice));
        impl.primaryRetained = true;

        // OptiX.
        RTCD_OPTIX_CHECK(initOptixOnce());
        OptixDeviceContextOptions optixOptions = {};
        if (options.optixValidation) {
            optixOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        }
        if (options.optixLogLevel > 0) {
            // OptiX levels: 1 fatal, 2 error, 3 warning, 4 print. Forwarded to the Context's logger.
            optixOptions.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void* data) {
                try {
                    const auto* self   = static_cast<const Impl*>(data);
                    const LogLevel l   = level <= 2 ? LogLevel::Error : level == 3 ? LogLevel::Warning : LogLevel::Info;
                    if (!self->options.logger.enabled(l)) {
                        return;
                    }
                    std::string text = "[optix][";
                    text += tag ? tag : "";
                    text += "] ";
                    text += message ? message : "";
                    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
                        text.pop_back();
                    }
                    self->options.logger.log(l, text);
                } catch (...) {
                    // never propagate into OptiX
                }
            };
            optixOptions.logCallbackData  = &impl;
            optixOptions.logCallbackLevel = static_cast<int>(options.optixLogLevel);
        }
        RTCD_OPTIX_CHECK(optixDeviceContextCreate(impl.cuContext, &optixOptions, &impl.optix));
        if (options.optixValidation) {
            impl.options.logger.log(LogLevel::Info, "Context: OptiX validation mode enabled");
        }
        return ctx;
    }

    Context::~Context() {
        if (!impl_) {
            return;
        }
        Impl& impl = *impl_;
        if (!impl.deviceSelected) {
            return; // create() failed before selecting the device: nothing was acquired
        }
        (void) cudaSetDevice(impl.info.ordinal);
        // No lease can be alive here (a lease holds a shared_ptr to its Context), so every stream is free.
        for (StreamHandle s : impl.allStreams) {
            const cudaError_t code = cudaStreamDestroy(s);
            if (code != cudaSuccess) {
                detail::logReleaseError("Context: cudaStreamDestroy", code, false);
            }
        }
        impl.allStreams.clear();
        impl.freeStreams.clear();
        if (impl.optix) {
            const OptixResult r = optixDeviceContextDestroy(impl.optix);
            if (r != OPTIX_SUCCESS) {
                try {
                    impl.options.logger.log(LogLevel::Error,
                        "Context: optixDeviceContextDestroy failed: " + detail::optixErrorText(static_cast<int>(r)));
                } catch (...) {
                }
            }
            impl.optix = nullptr;
        }
        if (impl.primaryRetained) {
            const CUresult r = cuDevicePrimaryCtxRelease(impl.cuDevice);
            if (r != CUDA_SUCCESS) {
                detail::logReleaseError("Context: cuDevicePrimaryCtxRelease", static_cast<int>(r), true);
            }
            impl.primaryRetained = false;
        }
    }

    int Context::device() const noexcept { return impl_->info.ordinal; }
    const DeviceInfo& Context::deviceInfo() const noexcept { return impl_->info; }
    CudaContextHandle Context::cudaContext() const noexcept { return impl_->cuContext; }
    OptixContextHandle Context::optixContext() const noexcept { return impl_->optix; }
    const Logger& Context::logger() const noexcept { return impl_->options.logger; }

    void Context::makeCurrent() const { RTCD_CUDA_CHECK(cudaSetDevice(impl_->info.ordinal)); }

    StreamLease Context::acquireStreams(std::uint32_t n) {
        if (n == 0) {
            throw InvalidArgument("Context::acquireStreams: n = 0, at least one stream must be leased");
        }
        makeCurrent();
        std::vector<StreamHandle> taken;
        taken.reserve(n);
        {
            std::lock_guard<std::mutex> lock(impl_->poolMutex);
            try {
                while (taken.size() < n && !impl_->freeStreams.empty()) {
                    taken.push_back(impl_->freeStreams.back());
                    impl_->freeStreams.pop_back();
                }
                while (taken.size() < n) {
                    cudaStream_t s = nullptr;
                    // Default flags (synchronize with the legacy default stream), as the old stack (Q13).
                    RTCD_CUDA_CHECK(cudaStreamCreate(&s));
                    impl_->allStreams.push_back(s);
                    taken.push_back(s);
                }
            } catch (...) {
                // Give back what was taken so a failed lease leaks nothing.
                impl_->freeStreams.insert(impl_->freeStreams.end(), taken.begin(), taken.end());
                throw;
            }
        }
        StreamLease lease;
        lease.owner_   = shared_from_this();
        lease.streams_ = std::move(taken);
        return lease;
    }

    void Context::returnStreams(std::vector<StreamHandle>& streams) noexcept {
        try {
            std::lock_guard<std::mutex> lock(impl_->poolMutex);
            impl_->freeStreams.insert(impl_->freeStreams.end(), streams.begin(), streams.end());
        } catch (...) {
            // allocation failure while returning: the streams stay owned by allStreams and are destroyed with the
            // Context; they are only lost for reuse
        }
        streams.clear();
    }

    // ---- StreamLease -------------------------------------------------------------------------------------------

    namespace {
        void syncStreamsNoThrow(const std::vector<StreamHandle>& streams) noexcept {
            for (StreamHandle s : streams) {
                const cudaError_t code = cudaStreamSynchronize(s);
                if (code != cudaSuccess) {
                    detail::logReleaseError("StreamLease: cudaStreamSynchronize", code, false);
                }
            }
        }
    } // namespace

    StreamLease::~StreamLease() {
        if (owner_) {
            syncStreamsNoThrow(streams_);
            owner_->returnStreams(streams_);
            owner_.reset();
        }
    }

    StreamLease::StreamLease(StreamLease&& other) noexcept
        : owner_(std::move(other.owner_)), streams_(std::move(other.streams_)) {
        other.owner_.reset();
        other.streams_.clear();
    }

    StreamLease& StreamLease::operator=(StreamLease&& other) noexcept {
        if (this != &other) {
            if (owner_) {
                syncStreamsNoThrow(streams_);
                owner_->returnStreams(streams_);
            }
            owner_   = std::move(other.owner_);
            streams_ = std::move(other.streams_);
            other.owner_.reset();
            other.streams_.clear();
        }
        return *this;
    }

} // namespace rtcd
