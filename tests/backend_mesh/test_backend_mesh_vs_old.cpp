// new_gpu, OLD-STACK test (deleted with the old stack): the mesh backend (through the ReferenceBatchRunner) gives
// the same per-item HIT COUNTS as the old CollisionDetector<MESH> of the IAS regression (side-by-side profile
// "reg_ias": NSTREAM 4, BATCHSIZE 128, COMPACT, useOBB) on the regression inputs, for the full run and for
// prefixes that exercise the masked path (partial batches) and the compaction path (full batches). Flags are the
// gate (test_backend_mesh); equal counts show that the same instances, rays and culling decisions are used.
#include "mesh_test_fixture.h"

#include <old_runner.h>

using namespace rtcd;
using rtcd::mesh_test::MeshBackendTest;

TEST_F(MeshBackendTest, HitCountsEqualTheOldIasDetector) {
    auto old = sbs::createOldRunner("reg_ias", ctx_->optixContext(), {});
    ASSERT_EQ(old->profile().batchSize, 128u);
    ASSERT_EQ(old->profile().streams, 4u);
    ASSERT_TRUE(old->profile().obb);
    ctx_->makeCurrent();
    auto h = makeHarness();
    for (const std::size_t n : {std::size_t(4096), std::size_t(517), std::size_t(37), std::size_t(1)}) {
        SCOPED_TRACE(n);
        old->upload(inputs_.data(), n);
        old->check();
        std::vector<std::uint32_t> oldCounts = old->hitCounts(n);
        ASSERT_GE(oldCounts.size(), n);
        oldCounts.resize(n); // the old runner may return its whole result buffer
        ctx_->makeCurrent();
        const std::vector<std::uint32_t> newCounts = h->runner->runAll(inputs_, static_cast<std::uint32_t>(n));
        ASSERT_EQ(newCounts.size(), oldCounts.size());
        std::size_t differing = 0, first = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (newCounts[i] != oldCounts[i]) {
                if (differing++ == 0) {
                    first = i;
                }
            }
        }
        EXPECT_EQ(differing, 0u) << "first differing item " << first << ": new " << (first < n ? newCounts[first] : 0)
                                 << ", old " << (first < n ? oldCounts[first] : 0);
    }
}
