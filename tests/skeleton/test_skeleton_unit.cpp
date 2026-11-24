// Skeleton smoke test (new_unit, no GPU): the embedded OptiX IR of rtcd_programs is present.
#include <rtcd_programs_embedded.h>

#include <gtest/gtest.h>

namespace rtcd::detail {
    void detectPlaceholder();
    void configPlaceholder();
} // namespace rtcd::detail

TEST(Skeleton, EmbeddedOptixIrIsPresent) {
    EXPECT_GT(rtcd::embedded::rtcd_programs_placeholder_size, 64u);
    EXPECT_NE(rtcd::embedded::rtcd_programs_placeholder[0] | rtcd::embedded::rtcd_programs_placeholder[1], 0);
}

TEST(Skeleton, ModulesLink) {
    rtcd::detail::detectPlaceholder();
    rtcd::detail::configPlaceholder();
    SUCCEED();
}
