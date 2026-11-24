// new_gpu: the mesh backend (through the ReferenceBatchRunner) gives the per-item HIT COUNTS the OLD
// CollisionDetector<MESH> of the IAS regression gave (stored: tests/data/reference/ias_hit_counts.txt / .bin,
// written from the old side-by-side profile "reg_ias": NSTREAM 4, BATCHSIZE 128, COMPACT, useOBB, before the old
// stack was removed), for the full run and for prefixes that exercise the masked path (partial batches) and the
// compaction path (full batches). Flags are the gate (test_backend_mesh); equal counts show that the same instances,
// rays and culling decisions are used.
#include "mesh_test_fixture.h"

#include <support/reference.h>

#include <numeric>

using namespace rtcd;
using rtcd::mesh_test::MeshBackendTest;

TEST_F(MeshBackendTest, HitCountsEqualTheStoredOldIasDetector) {
    const auto ref = support::ReferenceFile::load(support::referenceDir() / "ias_hit_counts.txt");
    const std::vector<std::uint32_t> full =
        [] {
            const auto f = support::readInt32File(support::referenceDir() / "ias_hit_counts.bin");
            return std::vector<std::uint32_t>(f.begin(), f.end());
        }();
    ASSERT_EQ(full.size(), 4096u);
    auto h = makeHarness();
    ASSERT_EQ(h->config.maxBatchSize, 128u);
    ASSERT_EQ(h->config.numStreams, 4u);
    ASSERT_TRUE(h->config.obbCullingEnabled());
    for (const std::size_t n : {std::size_t(4096), std::size_t(517), std::size_t(37), std::size_t(1)}) {
        SCOPED_TRACE(n);
        ctx_->makeCurrent();
        const std::vector<std::uint32_t> counts = h->runner->runAll(inputs_, static_cast<std::uint32_t>(n));
        ASSERT_EQ(counts.size(), n);
        const std::string key = std::to_string(n);
        EXPECT_EQ(ref.integer("hit_counts." + key, 0), n);
        EXPECT_EQ(support::hex64(support::fnv1a(counts)), support::hex64(ref.hash("hit_counts." + key, 1)));
        EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), std::uint64_t(0)), ref.integer("hits_total." + key));
        std::size_t differing = 0, first = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (counts[i] != full[i] && differing++ == 0) {
                first = i;
            }
        }
        // a prefix check gives the same counts as the full one (the stored prefixes say the old detector did too)
        EXPECT_EQ(differing, 0u) << "first differing item " << first << ": new " << (first < n ? counts[first] : 0)
                                 << ", stored " << (first < n ? full[first] : 0);
    }
}
