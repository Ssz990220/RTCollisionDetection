// rtcd_core (new_gpu): internal::obbOverlapCounts vs the stored counts of the OLD RTCD::BBCDRobotMask
// (tests/data/reference/obb_overlap.txt, written before the old stack was removed). Inputs: obb_reference_cases.h
// (deterministic random boxes incl. collapsed point boxes, and near-touching pairs right at the separation
// threshold). Per case: number of robot boxes with a non-zero count and the FNV-1a hash of all uint32 counts. Exact.
#include "obb_reference_cases.h"

#include <rtcd/context.h>
#include <rtcd/device_buffer.h>
#include <rtcd/internal/algorithms.h>

#include <support/reference.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

using namespace rtcd;
namespace sup = rtcd::support;

TEST(CoreObbReference, ObbOverlapCountsEqualTheStoredOldCounts) {
    const auto ref   = sup::ReferenceFile::load(sup::referenceDir() / "obb_overlap.txt");
    auto ctx         = Context::create();
    auto lease       = ctx->acquireStreams(1);
    const auto cases = obb_reference::cases();
    std::size_t checked = 0;
    for (const auto& c : cases) {
        SCOPED_TRACE(c.key);
        std::vector<std::uint32_t> all;
        for (const auto& r : c.runs) {
            DeviceBuffer dA = DeviceBuffer::fromHost(std::span<const Obb>(r.robot));
            DeviceBuffer dB = DeviceBuffer::fromHost(std::span<const Obb>(r.scene));
            DeviceBuffer counts(r.robot.size() * 4);
            counts.memsetAsync(0xAB, counts.size(), lease[0]);
            internal::obbOverlapCounts(dA.span<const Obb>(), dB.span<const Obb>(), counts.span<std::uint32_t>(), lease[0]);
            ASSERT_EQ(cudaStreamSynchronize(lease[0]), cudaSuccess);
            std::vector<std::uint32_t> h(r.robot.size());
            counts.download(h.data(), h.size() * 4);
            all.insert(all.end(), h.begin(), h.end());
        }
        std::size_t nonZero = 0;
        for (const auto v : all) nonZero += v != 0;
        EXPECT_EQ(nonZero, ref.integer(c.key, 0));
        EXPECT_EQ(sup::hex64(sup::fnv1a(all)), sup::hex64(ref.hash(c.key, 1)));
        ++checked;
    }
    EXPECT_EQ(checked, ref.keys().size()) << "every stored case is checked";
}
