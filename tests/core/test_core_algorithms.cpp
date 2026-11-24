// rtcd_core GPU tests (new_gpu): internal/algorithms.h against host references (scan, binarization, stable
// compaction, basic OBB overlap sanity). Bit-identity of obbOverlapCounts with the old BBCDRobotMask (stored counts)
// is in test_core_obb_reference.cpp.
#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>
#include <rtcd/internal/algorithms.h>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

using namespace rtcd;
using namespace rtcd::internal;

namespace {
    std::shared_ptr<Context> ctx() {
        static std::shared_ptr<Context> c = Context::create();
        return c;
    }

    struct Env {
        std::shared_ptr<Context> context = ctx();
        StreamLease lease                = context->acquireStreams(1);
        StreamHandle s() const { return lease[0]; }
        void sync() const { ASSERT_EQ(cudaStreamSynchronize(lease[0]), cudaSuccess); }
    };

    template <typename T>
    std::vector<T> download(const DeviceBuffer& b, std::size_t n) {
        std::vector<T> v(n);
        b.download(v.data(), n * sizeof(T));
        return v;
    }

    std::vector<std::uint32_t> randomU32(std::size_t n, std::mt19937& rng, std::uint32_t maxValue) {
        std::uniform_int_distribution<std::uint32_t> d(0, maxValue);
        std::vector<std::uint32_t> v(n);
        for (auto& x : v) {
            x = d(rng);
        }
        return v;
    }
} // namespace

// ---- exclusiveScanU32 ---------------------------------------------------------------------------------------------

TEST(CoreAlgorithms, ExclusiveScanMatchesHost) {
    Env env;
    std::mt19937 rng(1234);
    const std::size_t maxN = (1u << 20) + 3;
    DeviceBuffer scratch(exclusiveScanScratchBytes(maxN));
    DeviceBuffer in(maxN * 4), out(maxN * 4);
    for (std::size_t n : {1u, 2u, 3u, 4u, 5u, 31u, 100u, 1023u, 1024u, 1025u, 2048u, 4097u, 100000u, 1u << 20,
             (1u << 20) + 3}) {
        // large values so the uint32 sum wraps around
        auto h = randomU32(n, rng, n < 1000 ? 0xFFFFFFFFu : 1000u);
        in.upload(h.data(), n * 4);
        exclusiveScanU32(in.span<const std::uint32_t>().subspan(0, n), out.span<std::uint32_t>(),
            scratch.span<std::byte>(), env.s());
        env.sync();
        auto got = download<std::uint32_t>(out, n);
        std::uint32_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            ASSERT_EQ(got[i], acc) << "n = " << n << ", i = " << i;
            acc += h[i];
        }
        // in place (out aliases in)
        exclusiveScanU32(in.span<const std::uint32_t>().subspan(0, n), in.span<std::uint32_t>().subspan(0, n),
            scratch.span<std::byte>(), env.s());
        env.sync();
        EXPECT_EQ(download<std::uint32_t>(in, n), got) << "in-place, n = " << n;
    }
}

TEST(CoreAlgorithms, ExclusiveScanChecksArguments) {
    Env env;
    DeviceBuffer in(4096 * 4), out(4096 * 4);
    const std::size_t need = exclusiveScanScratchBytes(4096);
    ASSERT_GT(need, 1u);
    DeviceBuffer small(need - 1);
    std::vector<std::uint32_t> sentinel(4096, 0xABCDEF01u);
    out.upload(sentinel.data(), 4096 * 4);
    EXPECT_THROW(exclusiveScanU32(in.span<const std::uint32_t>(), out.span<std::uint32_t>(),
                     small.span<std::byte>(), env.s()),
        CapacityError);
    EXPECT_THROW(exclusiveScanU32(in.span<const std::uint32_t>(), out.span<std::uint32_t>().subspan(0, 10),
                     DeviceBuffer(need).span<std::byte>(), env.s()),
        InvalidArgument);
    env.sync();
    EXPECT_EQ(download<std::uint32_t>(out, 4096), sentinel); // nothing was enqueued
    // n = 0 is a no-op, scratch not needed
    EXPECT_NO_THROW(exclusiveScanU32({}, {}, {}, env.s()));
}

// ---- flagsToBinaryU32 ---------------------------------------------------------------------------------------------

TEST(CoreAlgorithms, FlagsToBinary) {
    Env env;
    std::mt19937 rng(7);
    for (std::size_t n : {1u, 255u, 256u, 257u, 100000u}) {
        auto h = randomU32(n, rng, 3);
        h[0]   = 0xFFFFFFFFu;
        DeviceBuffer in = DeviceBuffer::fromHost(std::span<const std::uint32_t>(h));
        DeviceBuffer out(n * 4);
        flagsToBinaryU32(in.span<const std::uint32_t>(), out.span<std::uint32_t>(), env.s());
        flagsToBinaryU32(in.span<const std::uint32_t>(), in.span<std::uint32_t>(), env.s()); // aliasing
        env.sync();
        auto a = download<std::uint32_t>(out, n);
        auto b = download<std::uint32_t>(in, n);
        for (std::size_t i = 0; i < n; ++i) {
            ASSERT_EQ(a[i], h[i] > 0 ? 1u : 0u);
            ASSERT_EQ(b[i], a[i]);
        }
    }
    DeviceBuffer x(16);
    EXPECT_THROW(flagsToBinaryU32(x.span<const std::uint32_t>(), x.span<std::uint32_t>().subspan(0, 3), nullptr),
        InvalidArgument);
}

// ---- compactByFlags -----------------------------------------------------------------------------------------------

namespace {
    // One compaction case: arrays with the given element sizes, random contents and flags (probability p), checked
    // against a host stable compaction. Bytes past the kept count must stay untouched.
    void runCompaction(const Env& env, std::size_t n, const std::vector<std::uint32_t>& elemBytes, double p,
        std::mt19937& rng) {
        std::bernoulli_distribution keep(p);
        std::uniform_int_distribution<std::uint32_t> flagValue(1, 1000);
        std::vector<std::uint32_t> flags(n);
        for (auto& f : flags) {
            f = keep(rng) ? flagValue(rng) : 0u;
        }
        DeviceBuffer dFlags = DeviceBuffer::fromHost(std::span<const std::uint32_t>(flags));

        std::vector<std::vector<std::uint32_t>> src(elemBytes.size());
        std::vector<DeviceBuffer> dSrc, dDst;
        std::vector<CompactArray> arrays;
        for (std::size_t k = 0; k < elemBytes.size(); ++k) {
            const std::size_t words = elemBytes[k] / 4;
            src[k]                  = randomU32(n * words, rng, 0xFFFFFFFFu);
            dSrc.push_back(DeviceBuffer::fromHost(std::span<const std::uint32_t>(src[k])));
            std::vector<std::uint32_t> fill(n * words, 0x5EED5EEDu);
            dDst.push_back(DeviceBuffer::fromHost(std::span<const std::uint32_t>(fill)));
            arrays.push_back({dDst[k].ptr(), dSrc[k].ptr(), elemBytes[k]});
        }
        std::uint32_t maxElem = 0;
        for (auto e : elemBytes) {
            maxElem = e > maxElem ? e : maxElem;
        }
        DeviceBuffer scratch(compactScratchBytes(n, maxElem));
        DeviceBuffer count(4);
        count.memsetAsync(0xFF, 4, env.s());
        compactByFlags(arrays, dFlags.span<const std::uint32_t>(), count.ptr(), scratch.span<std::byte>(), env.s());
        env.sync();

        std::uint32_t gotCount = 0;
        count.download(&gotCount, 4);
        std::vector<std::size_t> kept;
        for (std::size_t i = 0; i < n; ++i) {
            if (flags[i] > 0) {
                kept.push_back(i);
            }
        }
        ASSERT_EQ(gotCount, kept.size()) << "n = " << n << ", p = " << p;
        for (std::size_t k = 0; k < elemBytes.size(); ++k) {
            const std::size_t words = elemBytes[k] / 4;
            auto got                = download<std::uint32_t>(dDst[k], n * words);
            for (std::size_t j = 0; j < kept.size(); ++j) {
                for (std::size_t w = 0; w < words; ++w) {
                    ASSERT_EQ(got[j * words + w], src[k][kept[j] * words + w])
                        << "array " << k << ", kept element " << j << ", word " << w;
                }
            }
            for (std::size_t x = kept.size() * words; x < n * words; ++x) {
                ASSERT_EQ(got[x], 0x5EED5EEDu) << "array " << k << " written past the count";
            }
        }
    }
} // namespace

TEST(CoreAlgorithms, CompactByFlagsMatchesHostStableCompaction) {
    Env env;
    std::mt19937 rng(42);
    // 80 bytes = OptixInstance (mesh backend), 4 = instance -> item map, 12, 8: other layouts
    const std::vector<std::uint32_t> sizes = {80, 4, 12, 8};
    for (std::size_t n : {1u, 2u, 17u, 128u, 1000u, 128u * 8u, 1024u * 8u, 8192u * 17u}) {
        for (double p : {0.0, 0.05, 0.5, 1.0}) {
            runCompaction(env, n, sizes, p, rng);
            if (testing::Test::HasFatalFailure()) {
                return;
            }
        }
    }
    runCompaction(env, 3000, {4}, 0.3, rng);
}

TEST(CoreAlgorithms, CompactByFlagsEdgeCasesAndErrors) {
    Env env;
    DeviceBuffer count(4);
    // n = 0: count = 0, no scratch needed
    count.memsetAsync(0xFF, 4, env.s());
    EXPECT_NO_THROW(compactByFlags({}, {}, count.ptr(), {}, env.s()));
    env.sync();
    std::uint32_t c = 1;
    count.download(&c, 4);
    EXPECT_EQ(c, 0u);

    // no arrays: only the count
    std::vector<std::uint32_t> flags = {0, 3, 0, 1, 1};
    DeviceBuffer dFlags              = DeviceBuffer::fromHost(std::span<const std::uint32_t>(flags));
    DeviceBuffer scratch(compactScratchBytes(5, 4));
    compactByFlags({}, dFlags.span<const std::uint32_t>(), count.ptr(), scratch.span<std::byte>(), env.s());
    env.sync();
    count.download(&c, 4);
    EXPECT_EQ(c, 3u);

    DeviceBuffer a(5 * 8), b(5 * 8);
    const CompactArray ok{a.ptr(), b.ptr(), 8};
    const CompactArray five[5] = {ok, ok, ok, ok, ok};
    EXPECT_THROW(compactByFlags(five, dFlags.span<const std::uint32_t>(), count.ptr(), scratch.span<std::byte>(),
                     env.s()),
        InvalidArgument);
    const CompactArray odd{a.ptr(), b.ptr(), 6};
    EXPECT_THROW(compactByFlags({&odd, 1}, dFlags.span<const std::uint32_t>(), count.ptr(), scratch.span<std::byte>(),
                     env.s()),
        InvalidArgument);
    const CompactArray overlap{a.ptr(), a.ptr() + 8, 8};
    EXPECT_THROW(compactByFlags({&overlap, 1}, dFlags.span<const std::uint32_t>(), count.ptr(),
                     scratch.span<std::byte>(), env.s()),
        InvalidArgument);
    EXPECT_THROW(compactByFlags({&ok, 1}, dFlags.span<const std::uint32_t>(), 0, scratch.span<std::byte>(), env.s()),
        InvalidArgument);
    // too little scratch: CapacityError, nothing enqueued (count keeps its value)
    count.memsetAsync(0x77, 4, env.s());
    env.sync();
    EXPECT_THROW(compactByFlags({&ok, 1}, dFlags.span<const std::uint32_t>(), count.ptr(),
                     scratch.span<std::byte>().subspan(0, scratch.size() - 1), env.s()),
        CapacityError);
    env.sync();
    count.download(&c, 4);
    EXPECT_EQ(c, 0x77777777u);
    // scratch sized for a larger maximum is fine for a smaller call
    EXPECT_GE(compactScratchBytes(8192, 80), compactScratchBytes(5, 80));
}

// ---- obbOverlapCounts sanity (exact parity with the old kernel: test_core_obb_reference) ------------------------

TEST(CoreAlgorithms, ObbOverlapCountsSanity) {
    Env env;
    auto box = [](float x, float y, float z, float h) {
        Obb o{};
        o.center[0] = x, o.center[1] = y, o.center[2] = z;
        o.halfSize[0] = o.halfSize[1] = o.halfSize[2] = h;
        o.axes[0][0] = o.axes[1][1] = o.axes[2][2] = 1.0f;
        return o;
    };
    // a: 40 boxes along x; b: two boxes at x = 0 and x = 10
    std::vector<Obb> a, b = {box(0, 0, 0, 1), box(10, 0, 0, 1)};
    for (int i = 0; i < 40; ++i) {
        a.push_back(box(float(i) * 0.5f, 0, 0, 0.2f));
    }
    a.push_back(box(5, 0, 0, 0)); // a point (collapsed root OBB): overlaps nothing here
    a.push_back(box(0, 0, 0, 0)); // a point inside b[0]: still overlaps
    DeviceBuffer dA = DeviceBuffer::fromHost(std::span<const Obb>(a));
    DeviceBuffer dB = DeviceBuffer::fromHost(std::span<const Obb>(b));
    DeviceBuffer counts(a.size() * 4);
    counts.memsetAsync(0xFF, counts.size(), env.s()); // no pre-zeroing needed
    obbOverlapCounts(dA.span<const Obb>(), dB.span<const Obb>(), counts.span<std::uint32_t>(), env.s());
    env.sync();
    auto got = download<std::uint32_t>(counts, a.size());
    for (int i = 0; i < 40; ++i) {
        const float x       = float(i) * 0.5f;
        const unsigned want = (x <= 1.2f ? 1u : 0u) + (x >= 8.8f && x <= 11.2f ? 1u : 0u);
        EXPECT_EQ(got[i], want) << "x = " << x;
    }
    EXPECT_EQ(got[40], 0u);
    EXPECT_EQ(got[41], 1u);

    // empty b: all zero
    obbOverlapCounts(dA.span<const Obb>(), {}, counts.span<std::uint32_t>(), env.s());
    env.sync();
    for (auto v : download<std::uint32_t>(counts, a.size())) {
        EXPECT_EQ(v, 0u);
    }
    EXPECT_THROW(obbOverlapCounts(dA.span<const Obb>(), dB.span<const Obb>(),
                     counts.span<std::uint32_t>().subspan(0, 3), env.s()),
        InvalidArgument);
    EXPECT_NO_THROW(obbOverlapCounts({}, dB.span<const Obb>(), {}, env.s()));
}
