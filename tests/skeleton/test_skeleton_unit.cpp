// Skeleton smoke test (new_unit, no GPU): the OptiX IR embedded by rtcd_add_optix_ir is present, and every
// library of the new tree links.
#include <rtcd_skeleton_ir_embedded.h>

#include <gtest/gtest.h>

namespace rtcd::detail {
    void corePlaceholder();
    void configPlaceholder();
    void jsonPlaceholder();
    void geometryPlaceholder();
    void scenePlaceholder();
    void robotPlaceholder();
    void backendMeshPlaceholder();
    void backendRayPlaceholder();
    void backendCurvePlaceholder();
    void detectorPlaceholder();
    void testsupportPlaceholder();
} // namespace rtcd::detail

TEST(Skeleton, EmbeddedOptixIrIsPresent) {
    EXPECT_GT(rtcd::embedded::rtcd_skeleton_ir_skeleton_program_size, 64u);
    EXPECT_NE(rtcd::embedded::rtcd_skeleton_ir_skeleton_program[0]
                  | rtcd::embedded::rtcd_skeleton_ir_skeleton_program[1],
        0);
}

TEST(Skeleton, ModulesLink) {
    rtcd::detail::corePlaceholder();
    rtcd::detail::configPlaceholder();
    rtcd::detail::jsonPlaceholder();
    rtcd::detail::geometryPlaceholder();
    rtcd::detail::scenePlaceholder();
    rtcd::detail::robotPlaceholder();
    rtcd::detail::backendMeshPlaceholder();
    rtcd::detail::backendRayPlaceholder();
    rtcd::detail::backendCurvePlaceholder();
    rtcd::detail::detectorPlaceholder();
    rtcd::detail::testsupportPlaceholder();
    SUCCEED();
}
