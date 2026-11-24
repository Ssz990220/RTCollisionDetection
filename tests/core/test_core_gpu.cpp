// rtcd_core GPU tests (new_gpu): Context, StreamLease, DeviceBuffer, PinnedHostBuffer, real CUDA error
// classification.
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_types.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace rtcd_test {
    void launchIllegalAddressKernel();
    void launchInvalidConfiguration();
} // namespace rtcd_test

namespace {
    bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

    std::shared_ptr<rtcd::Context> sharedContext() {
        static std::shared_ptr<rtcd::Context> ctx = rtcd::Context::create();
        return ctx;
    }
} // namespace

// ---- Context -----------------------------------------------------------------------------------------------------

TEST(CoreContext, CreateReportsDeviceAndHandles) {
    auto ctx = sharedContext();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->device(), 0);
    EXPECT_EQ(ctx->deviceInfo().ordinal, 0);
    EXPECT_FALSE(ctx->deviceInfo().name.empty());
    EXPECT_GE(ctx->deviceInfo().computeMajor, 5);
    EXPECT_GT(ctx->deviceInfo().multiprocessors, 0);
    EXPECT_GT(ctx->deviceInfo().globalMemory, 0u);
    EXPECT_NE(ctx->cudaContext(), nullptr);
    EXPECT_NE(ctx->optixContext(), nullptr);
    EXPECT_NO_THROW(ctx->makeCurrent());

    // the CUDA context is the device's primary context, current after makeCurrent()
    CUcontext current = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
    EXPECT_EQ(current, ctx->cudaContext());
}

TEST(CoreContext, SeveralContextsOnOneDevice) {
    auto a = rtcd::Context::create();
    auto b = rtcd::Context::create();
    EXPECT_NE(a->optixContext(), b->optixContext());
    EXPECT_EQ(a->cudaContext(), b->cudaContext()); // same primary context
    a.reset();
    // b still works after a is gone (a released only its own retain of the primary context)
    rtcd::DeviceBuffer buf(64);
    auto lease = b->acquireStreams(1);
    EXPECT_NO_THROW(buf.memsetAsync(0, 64, lease[0]));
    EXPECT_EQ(cudaStreamSynchronize(lease[0]), cudaSuccess);
}

TEST(CoreContext, InvalidOptionsAndDevice) {
    rtcd::ContextOptions o;
    o.optixLogLevel = 5;
    EXPECT_THROW(rtcd::Context::create(o), rtcd::InvalidArgument);
    o.optixLogLevel = 0;
    o.device        = -1;
    EXPECT_THROW(rtcd::Context::create(o), rtcd::InvalidArgument);
    o.device = 999;
    try {
        rtcd::Context::create(o);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_EQ(e.code(), int(cudaErrorInvalidDevice));
        EXPECT_FALSE(e.sticky());
    }
    // the failure is not left behind as the runtime's last error
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_NO_THROW(sharedContext()->makeCurrent());
}

TEST(CoreContext, OptixLogIsForwardedToLogger) {
    std::mutex m;
    std::vector<std::string> lines;
    rtcd::ContextOptions o;
    o.optixValidation = true;
    o.optixLogLevel   = 4;
    o.logger.level    = rtcd::LogLevel::Debug;
    o.logger.sink     = [&](rtcd::LogLevel, std::string_view msg) {
        std::lock_guard<std::mutex> lock(m);
        lines.emplace_back(msg);
    };
    auto ctx = rtcd::Context::create(o);
    std::lock_guard<std::mutex> lock(m);
    ASSERT_FALSE(lines.empty()) << "expected at least one OptiX log message at level 4 with validation on";
    bool anyOptix = false;
    for (const auto& l : lines) {
        anyOptix = anyOptix || contains(l, "[optix][");
    }
    EXPECT_TRUE(anyOptix);
}

TEST(CoreContext, OptixErrorNamesAfterInit) {
    sharedContext(); // optixInit() has run
    try {
        RTCD_OPTIX_CHECK(OPTIX_ERROR_INVALID_VALUE);
        FAIL() << "no throw";
    } catch (const rtcd::OptixError& e) {
        EXPECT_TRUE(contains(e.what(), "OPTIX_ERROR_INVALID_VALUE")) << e.what();
    }
    try {
        RTCD_CU_CHECK(CUDA_ERROR_LAUNCH_FAILED);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_TRUE(contains(e.what(), "CUDA_ERROR_LAUNCH_FAILED")) << e.what();
    }
}

// ---- StreamLease -------------------------------------------------------------------------------------------------

TEST(CoreStreams, LeasesAreExclusiveAndReused) {
    auto ctx = rtcd::Context::create();
    EXPECT_THROW(ctx->acquireStreams(0), rtcd::InvalidArgument);

    std::set<rtcd::StreamHandle> first;
    {
        auto a = ctx->acquireStreams(4);
        ASSERT_EQ(a.size(), 4u);
        first.insert(a.streams().begin(), a.streams().end());
        EXPECT_EQ(first.size(), 4u);
        EXPECT_EQ(first.count(nullptr), 0u); // never the legacy default stream
        auto b = ctx->acquireStreams(3);
        for (auto s : b.streams()) {
            EXPECT_EQ(first.count(s), 0u) << "stream leased twice";
        }
        EXPECT_THROW((void) a[4], std::out_of_range);
    }
    // all 7 are back; a new lease of 4 reuses pool streams (no new ones needed)
    auto c = ctx->acquireStreams(7);
    std::set<rtcd::StreamHandle> again(c.streams().begin(), c.streams().end());
    EXPECT_EQ(again.size(), 7u);
    for (auto s : first) {
        EXPECT_EQ(again.count(s), 1u);
    }
}

TEST(CoreStreams, MoveAndDestructionSynchronize) {
    auto ctx = rtcd::Context::create();
    rtcd::DeviceBuffer big(64u << 20);
    rtcd::StreamHandle s = nullptr;
    {
        auto a = ctx->acquireStreams(1);
        s      = a[0];
        rtcd::StreamLease b(std::move(a));
        EXPECT_EQ(a.size(), 0u);
        EXPECT_EQ(b.size(), 1u);
        rtcd::StreamLease c;
        c = std::move(b);
        EXPECT_EQ(b.size(), 0u);
        ASSERT_EQ(c[0], s);
        for (int i = 0; i < 8; ++i) {
            big.memsetAsync(i, big.size(), c[0]);
        }
    } // lease destroyed: it synchronized the stream before returning it
    EXPECT_EQ(cudaStreamQuery(s), cudaSuccess);
    auto again = ctx->acquireStreams(1);
    EXPECT_EQ(again[0], s);
}

TEST(CoreStreams, ConcurrentLeasesNeverShareStreams) {
    auto ctx = rtcd::Context::create();
    std::mutex m;
    std::multiset<rtcd::StreamHandle> held;
    std::atomic<int> violations{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) {
                auto lease = ctx->acquireStreams(2);
                {
                    std::lock_guard<std::mutex> lock(m);
                    for (auto s : lease.streams()) {
                        if (held.count(s) != 0) {
                            ++violations;
                        }
                        held.insert(s);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(m);
                    for (auto s : lease.streams()) {
                        held.erase(held.find(s));
                    }
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(violations.load(), 0);
}

// ---- DeviceBuffer ------------------------------------------------------------------------------------------------

TEST(CoreDeviceBuffer, AllocateUploadDownload) {
    sharedContext()->makeCurrent();
    rtcd::DeviceBuffer empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.ptr(), 0u);

    std::vector<int> host(1000);
    std::iota(host.begin(), host.end(), -500);
    rtcd::DeviceBuffer buf(host.size() * sizeof(int));
    EXPECT_EQ(buf.size(), 4000u);
    EXPECT_NE(buf.ptr(), 0u);
    buf.upload(host.data(), 4000);
    std::vector<int> back(1000, 0);
    buf.download(back.data(), 4000);
    EXPECT_EQ(back, host);

    // offsets
    const int seven[2] = {7, 7};
    buf.upload(seven, sizeof(seven), 40);
    int two[3] = {};
    buf.download(two, sizeof(two), 36);
    EXPECT_EQ(two[0], host[9]);
    EXPECT_EQ(two[1], 7);
    EXPECT_EQ(two[2], 7);

    // typed helpers
    auto span = buf.span<int>();
    EXPECT_EQ(span.count, 1000u);
    EXPECT_EQ(span.ptr, buf.ptr());
    auto fromHost = rtcd::DeviceBuffer::fromHost(std::span<const int>(host));
    EXPECT_EQ(fromHost.size(), 4000u);
    std::vector<int> back2(1000);
    fromHost.download(back2.data(), 4000);
    EXPECT_EQ(back2, host);
    auto fromEmpty = rtcd::DeviceBuffer::fromHost(std::span<const int>());
    EXPECT_TRUE(fromEmpty.empty());
}

TEST(CoreDeviceBuffer, RangeChecksThrowBeforeTouchingCuda) {
    sharedContext()->makeCurrent();
    rtcd::DeviceBuffer buf(16);
    const char src[32] = {};
    char dst[32]       = {};
    EXPECT_THROW(buf.upload(src, 17), rtcd::InvalidArgument);
    EXPECT_THROW(buf.upload(src, 8, 9), rtcd::InvalidArgument);
    EXPECT_THROW(buf.upload(src, 1, ~std::size_t(0)), rtcd::InvalidArgument); // no overflow wrap
    EXPECT_THROW(buf.download(dst, 2, 15), rtcd::InvalidArgument);
    EXPECT_THROW(buf.upload(nullptr, 4), rtcd::InvalidArgument);
    EXPECT_THROW(buf.download(nullptr, 4), rtcd::InvalidArgument);
    EXPECT_THROW(buf.uploadAsync(src, 17, nullptr), rtcd::InvalidArgument);
    EXPECT_THROW(buf.downloadAsync(dst, 17, nullptr), rtcd::InvalidArgument);
    EXPECT_THROW(buf.copyFromAsync(buf.ptr(), 17, nullptr), rtcd::InvalidArgument);
    EXPECT_THROW(buf.copyFromAsync(0, 4, nullptr), rtcd::InvalidArgument);
    EXPECT_THROW(buf.memsetAsync(0, 1, nullptr, 16), rtcd::InvalidArgument);
    // boundary cases are fine
    EXPECT_NO_THROW(buf.upload(src, 16));
    EXPECT_NO_THROW(buf.upload(src, 0, 16));
    EXPECT_NO_THROW(buf.memsetAsync(0, 0, nullptr, 16));
    rtcd::DeviceBuffer none;
    EXPECT_NO_THROW(none.upload(nullptr, 0));
    EXPECT_THROW(none.upload(src, 1), rtcd::InvalidArgument);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(CoreDeviceBuffer, AllocateReserveReleaseMove) {
    sharedContext()->makeCurrent();
    rtcd::DeviceBuffer buf(256);
    const auto p = buf.ptr();
    buf.reserve(128); // no-op: big enough
    EXPECT_EQ(buf.ptr(), p);
    EXPECT_EQ(buf.size(), 256u);
    buf.reserve(1024);
    EXPECT_EQ(buf.size(), 1024u);
    buf.allocate(0);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.ptr(), 0u);
    buf.allocate(512);
    EXPECT_EQ(buf.size(), 512u);

    const auto q = buf.ptr();
    rtcd::DeviceBuffer moved(std::move(buf));
    EXPECT_EQ(moved.ptr(), q);
    EXPECT_EQ(moved.size(), 512u);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.ptr(), 0u);

    rtcd::DeviceBuffer other(64);
    other = std::move(moved); // releases other's 64 bytes first
    EXPECT_EQ(other.ptr(), q);
    EXPECT_TRUE(moved.empty());
    auto& self = other;
    other      = std::move(self); // self-move keeps the allocation
    EXPECT_EQ(other.ptr(), q);
    other.release();
    EXPECT_TRUE(other.empty());
    other.release(); // idempotent
}

TEST(CoreDeviceBuffer, AsyncCopiesWithPinnedMemory) {
    auto ctx   = sharedContext();
    auto lease = ctx->acquireStreams(1);
    auto s     = lease[0];

    rtcd::PinnedHostBuffer pin(1024 * sizeof(float));
    EXPECT_EQ(pin.size(), 4096u);
    auto in = pin.as<float>();
    ASSERT_EQ(in.size(), 1024u);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = float(i) * 0.5f;
    }
    rtcd::DeviceBuffer a(4096), b(4096);
    a.uploadAsync(pin.data(), 4096, s);
    b.memsetAsync(0, 4096, s);
    b.copyFromAsync(a.ptr() + 1024, 2048, s, 0); // b[0..512) = a[256..768)
    b.memsetAsync(0xFF, 16, s, 4080);             // last 4 floats = NaN pattern
    rtcd::PinnedHostBuffer out(4096);
    b.downloadAsync(out.data(), 4096, s);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    auto o = out.as<float>();
    for (std::size_t i = 0; i < 512; ++i) {
        ASSERT_EQ(o[i], float(i + 256) * 0.5f) << i;
    }
    for (std::size_t i = 512; i < 1020; ++i) {
        ASSERT_EQ(o[i], 0.0f) << i;
    }
    const auto* raw = static_cast<const unsigned char*>(out.data());
    for (std::size_t i = 4080; i < 4096; ++i) {
        ASSERT_EQ(raw[i], 0xFF);
    }

    // pageable host memory also works with the async variants
    std::vector<float> pageable(1024, 3.0f);
    a.uploadAsync(pageable.data(), 4096, s);
    std::vector<float> back(1024, 0.0f);
    a.downloadAsync(back.data(), 4096, s, 0);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    EXPECT_EQ(back, pageable);
}

TEST(CoreDeviceBuffer, PinnedMove) {
    sharedContext()->makeCurrent();
    rtcd::PinnedHostBuffer a(64);
    void* p = a.data();
    rtcd::PinnedHostBuffer b(std::move(a));
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
    rtcd::PinnedHostBuffer c(8);
    c = std::move(b);
    EXPECT_EQ(c.data(), p);
    EXPECT_EQ(c.size(), 64u);
    c.allocate(0);
    EXPECT_EQ(c.data(), nullptr);
}

TEST(CoreDeviceBuffer, AllocationFailureNamesTheSizeAndIsRecoverable) {
    sharedContext()->makeCurrent();
    const std::size_t huge = std::size_t(1) << 50; // 1 PiB
    try {
        rtcd::DeviceBuffer b(huge);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_EQ(e.code(), int(cudaErrorMemoryAllocation));
        EXPECT_FALSE(e.sticky());
        EXPECT_TRUE(contains(e.what(), std::to_string(huge))) << e.what();
    }
    EXPECT_EQ(cudaGetLastError(), cudaSuccess); // not left as the last error
    rtcd::DeviceBuffer ok(1024);
    EXPECT_FALSE(ok.empty());

    rtcd::DeviceBuffer keep(32);
    EXPECT_THROW(keep.allocate(huge), rtcd::CudaError);
    EXPECT_TRUE(keep.empty()); // allocate() discards the old contents first, also on failure
}

// ---- real CUDA failures --------------------------------------------------------------------------------------------

TEST(CoreErrorsGpu, LaunchConfigurationErrorIsNotSticky) {
    sharedContext()->makeCurrent();
    rtcd_test::launchInvalidConfiguration();
    try {
        RTCD_CUDA_CHECK(cudaGetLastError());
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        // CUDA 13.2 reports 1 << 20 threads per block as cudaErrorInvalidValue (older toolkits:
        // cudaErrorInvalidConfiguration); either way a launch-configuration error, not sticky
        EXPECT_TRUE(e.code() == int(cudaErrorInvalidValue) || e.code() == int(cudaErrorInvalidConfiguration))
            << e.code();
        EXPECT_FALSE(e.sticky());
    }
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    rtcd::DeviceBuffer stillWorks(16);
    EXPECT_NO_THROW(stillWorks.memsetAsync(0, 16, nullptr));
}

// The fault kills the CUDA context of the process, so it runs in a child process (gtest death test).
TEST(CoreErrorsGpuDeathTest, DeviceFaultSurfacesAsStickyErrorAtNextSync) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            auto ctx   = rtcd::Context::create();
            auto lease = ctx->acquireStreams(1);
            int status = 1;
            rtcd_test::launchIllegalAddressKernel();
            try {
                RTCD_CUDA_CHECK(cudaGetLastError()); // the launch itself succeeds
                status = 2;
                RTCD_CUDA_CHECK(cudaDeviceSynchronize());
                status = 3;
            } catch (const rtcd::CudaError& e) {
                if (status == 2 && e.sticky() && e.code() == int(cudaErrorIllegalAddress)) {
                    status = 0;
                    // every later call reports the sticky error too
                    try {
                        rtcd::DeviceBuffer b(16);
                        status = 4;
                    } catch (const rtcd::CudaError& e2) {
                        status = e2.sticky() ? 42 : 5;
                    }
                }
            }
            std::_Exit(status);
        },
        testing::ExitedWithCode(42), "");
}
