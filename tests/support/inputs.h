#pragma once
// support/inputs.h -- the deterministic collision-check inputs cached under tests/data/motions (committed; see
// tests/data/README.md; <motions> below is dataDir()/motions), read exactly as the old
// harness reads them (RTCD/Utils/Test/poseUtils.h randomMotions / KNearestNeighbor / genCSpaceKNNTraj and
// RTCD/CollisionDetector/Test/ccRegression.cpp loadInputs), so the new stack is checked and timed on
// byte-identical inputs.
//
// Cache layout (old naming, unchanged):
//   <motions>/<robot><n>.bin                               n poses, dof float32 each (old randomMotions)
//   <motions>/<robot>/<weights>/<n>_knn<k>_idx.bin         n * k int32 neighbour indices (old KNearestNeighbor)
//   <motions>/<robot>/<weights>/<n>_knn<k>_dis.bin         n * n float32 distances (never read; not committed)
// <robot> is the OLD robot name ("Panda", RTCD/Robot/models/pandas.h Config), <weights> the old
// concatFloatsAsInts(weight) ("1111111" for the unit weights of the regression and the benchmarks).
//
// The old harness generated a missing cache (Halton samples within the pandas.h bounds, GPU KNN). This module only
// reads: a missing or malformed cache file throws rtcd::IoError (design doc Q7: inputs stay identical; the only
// generator is the removed old stack, tag old-stack-final). The flat output layout is the item layout (rtcd/types.h):
// [item][waypoint][joint], packed.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rtcd::support {

    // Constants of the old harness (ccRegression.cpp, RTCD/Benchmark/*). They name cache files; they are not
    // properties of the robot model.
    struct LegacyHarness {
        static constexpr std::string_view robotName = "Panda"; // old robotConfig name (cache file prefix)
        static constexpr std::string_view weightString = "1111111"; // concatFloatsAsInts({1,1,1,1,1,1,1})
        static constexpr std::uint32_t dof               = 7;
        static constexpr std::uint32_t knnK              = 4;
        static constexpr std::size_t regressionItems     = 4096; // NITEMS of ccRegression.cpp
        static constexpr std::size_t benchmarkSamples    = 8192; // trajLength (discrete) / nCSpaceSample (curve)
    };

    // old concatFloatsAsInts: every weight truncated to int and printed, concatenated.
    std::string knnWeightString(std::span<const float> weights);

    // <motionsDir>/<robotName><count>.bin: `count` poses of `dof` floats (count * dof floats). Throws IoError if the
    // file is missing or does not hold exactly `count` whole poses (old randomMotions size check; a trailing
    // partial pose is ignored like the old readBin).
    std::vector<float> loadCachedPoses(const std::filesystem::path& motionsDir, std::string_view robotName,
        std::size_t count, std::uint32_t dof);

    // <motionsDir>/<robotName>/<weightString>/<samples>_knn<k>_idx.bin: samples * k int32 indices, each in
    // [0, samples). Throws IoError on a missing file, a wrong size or an out-of-range index.
    std::vector<std::int32_t> loadKnnIndices(const std::filesystem::path& motionsDir, std::string_view robotName,
        std::string_view weightString, std::size_t samples, std::uint32_t k);

    // Straight-line trajectories from every sample to each of its k nearest neighbours, exactly the float
    // arithmetic of the old genCSpaceKNNTraj: trajectory t = i * k + j (sample i, neighbour j), waypoint p:
    //   span  = (end - start) / float(trajectoryPoints - 1)      (per joint, float)
    //   value = start + span * float(p)                          (per joint, float, product rounded first)
    // Only the first `maxTrajectories` trajectories are produced (all if maxTrajectories >= samples * k).
    // Output: [trajectory][waypoint][joint] floats. Throws InvalidArgument on inconsistent sizes.
    std::vector<float> knnTrajectories(std::span<const float> samples, std::uint32_t dof,
        std::span<const std::int32_t> knnIndices, std::uint32_t k, std::uint32_t trajectoryPoints,
        std::size_t maxTrajectories);

    // The regression inputs of ccRegression.cpp with `items` = NITEMS (4096 for the goldens):
    //   pointsPerItem == 1 (discrete): the first `items` poses of <robot><items>.bin;
    //   pointsPerItem >= 2 (curves):   the first `items` KNN trajectories (k = 4, unit weights) with
    //                                  pointsPerItem waypoints over the <items> cached samples.
    std::vector<float> regressionInputs(const std::filesystem::path& motionsDir, std::uint32_t pointsPerItem,
        std::size_t items = LegacyHarness::regressionItems, std::uint32_t dof = LegacyHarness::dof,
        std::string_view robotName = LegacyHarness::robotName);

    // Benchmark pools of the old benchmarks (8192 cached samples):
    //   discrete (RTCD/Benchmark/Discrete): the 8192 poses;
    //   curves   (RTCD/Benchmark/Curve):    all 8192 * 4 KNN trajectories with `trajectoryPoints` waypoints.
    std::vector<float> benchmarkPosePool(const std::filesystem::path& motionsDir,
        std::uint32_t dof = LegacyHarness::dof, std::string_view robotName = LegacyHarness::robotName);
    std::vector<float> benchmarkTrajectoryPool(const std::filesystem::path& motionsDir, std::uint32_t trajectoryPoints,
        std::uint32_t dof = LegacyHarness::dof, std::string_view robotName = LegacyHarness::robotName);

    // Raw little-endian file reads (whole elements; a trailing partial element is ignored, like the old readBin).
    std::vector<float> readFloatFile(const std::filesystem::path& file);
    std::vector<std::int32_t> readInt32File(const std::filesystem::path& file);

} // namespace rtcd::support
