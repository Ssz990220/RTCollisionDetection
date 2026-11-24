// rtcd_core unit tests (new_unit, no GPU needed): checking macros, exception classification, Logger.
#include <rtcd/errors.h>
#include <rtcd/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_types.h>

#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <vector>

namespace {
    bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

    int g_calls = 0;
    int countedCall(int result) {
        ++g_calls;
        return result;
    }
} // namespace

TEST(CoreErrors, SuccessDoesNotThrowAndEvaluatesOnce) {
    g_calls = 0;
    EXPECT_NO_THROW(RTCD_CUDA_CHECK(countedCall(0)));
    EXPECT_NO_THROW(RTCD_CU_CHECK(countedCall(0)));
    EXPECT_NO_THROW(RTCD_OPTIX_CHECK(countedCall(0)));
    EXPECT_NO_THROW(RTCD_OPTIX_CHECK_LOG(countedCall(0), "log"));
    EXPECT_EQ(g_calls, 4);
    g_calls = 0;
    EXPECT_THROW(RTCD_CUDA_CHECK(countedCall(int(cudaErrorInvalidValue))), rtcd::CudaError);
    EXPECT_EQ(g_calls, 1);
}

TEST(CoreErrors, CallerVariableNamedResIsNotShadowed) {
    // the old OPTIX_CHECK declared `res` itself; the new macros use a reserved name
    const int res = 0;
    EXPECT_NO_THROW(RTCD_CUDA_CHECK(res));
    EXPECT_NO_THROW(RTCD_OPTIX_CHECK(res));
    // nesting compiles and works
    EXPECT_NO_THROW(RTCD_CUDA_CHECK([] {
        RTCD_CUDA_CHECK(0);
        return 0;
    }()));
}

TEST(CoreErrors, CudaRuntimeErrorIsClassified) {
    try {
        RTCD_CUDA_CHECK(cudaErrorMemoryAllocation);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_EQ(e.kind(), rtcd::ErrorKind::Cuda);
        EXPECT_EQ(e.code(), int(cudaErrorMemoryAllocation));
        EXPECT_FALSE(e.driver());
        EXPECT_FALSE(e.sticky());
        const std::string w = e.what();
        EXPECT_TRUE(contains(w, "cudaErrorMemoryAllocation")) << w;
        EXPECT_TRUE(contains(w, "test_core_unit.cpp")) << w;
        EXPECT_TRUE(contains(w, "RTCD_CUDA_CHECK") || contains(w, "cudaErrorMemoryAllocation failed")) << w;
    }
}

TEST(CoreErrors, StickyRuntimeErrors) {
    const cudaError_t sticky[] = {cudaErrorIllegalAddress, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
        cudaErrorAssert, cudaErrorHardwareStackError, cudaErrorIllegalInstruction, cudaErrorMisalignedAddress,
        cudaErrorInvalidAddressSpace, cudaErrorInvalidPc, cudaErrorECCUncorrectable};
    for (cudaError_t c : sticky) {
        try {
            RTCD_CUDA_CHECK(c);
            FAIL() << "no throw";
        } catch (const rtcd::CudaError& e) {
            EXPECT_TRUE(e.sticky()) << int(c);
            EXPECT_TRUE(contains(e.what(), "sticky")) << e.what();
        }
    }
    const cudaError_t notSticky[] = {cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorInvalidConfiguration,
        cudaErrorLaunchOutOfResources, cudaErrorInvalidDevice, cudaErrorNoDevice, cudaErrorNotReady};
    for (cudaError_t c : notSticky) {
        try {
            RTCD_CUDA_CHECK(c);
            FAIL() << "no throw";
        } catch (const rtcd::CudaError& e) {
            EXPECT_FALSE(e.sticky()) << int(c);
        }
    }
}

TEST(CoreErrors, DriverErrorIsClassified) {
    try {
        RTCD_CU_CHECK(CUDA_ERROR_LAUNCH_FAILED);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_EQ(e.code(), int(CUDA_ERROR_LAUNCH_FAILED));
        EXPECT_TRUE(e.driver());
        EXPECT_TRUE(e.sticky());
    }
    try {
        RTCD_CU_CHECK(CUDA_ERROR_INVALID_VALUE);
        FAIL() << "no throw";
    } catch (const rtcd::CudaError& e) {
        EXPECT_TRUE(e.driver());
        EXPECT_FALSE(e.sticky());
    }
}

TEST(CoreErrors, OptixErrorBeforeInitAndLog) {
    // This executable never calls optixInit(): the helper must not touch the empty function table.
    try {
        RTCD_OPTIX_CHECK(OPTIX_ERROR_INVALID_VALUE);
        FAIL() << "no throw";
    } catch (const rtcd::OptixError& e) {
        EXPECT_EQ(e.kind(), rtcd::ErrorKind::Optix);
        EXPECT_EQ(e.code(), int(OPTIX_ERROR_INVALID_VALUE));
        EXPECT_TRUE(e.log().empty());
        EXPECT_TRUE(contains(e.what(), std::to_string(int(OPTIX_ERROR_INVALID_VALUE)))) << e.what();
    }
    const char log[] = "error: something did not compile";
    try {
        RTCD_OPTIX_CHECK_LOG(OPTIX_ERROR_PIPELINE_LINK_ERROR, log);
        FAIL() << "no throw";
    } catch (const rtcd::OptixError& e) {
        EXPECT_EQ(e.log(), log);
        EXPECT_TRUE(contains(e.what(), log)) << e.what();
    }
    // an empty log is not attached
    try {
        RTCD_OPTIX_CHECK_LOG(OPTIX_ERROR_PIPELINE_LINK_ERROR, "");
        FAIL() << "no throw";
    } catch (const rtcd::OptixError& e) {
        EXPECT_TRUE(e.log().empty());
    }
}

TEST(CoreErrors, AllErrorsDeriveFromRtcdError) {
    EXPECT_THROW(throw rtcd::ConfigError("x", "a.b"), rtcd::Error);
    EXPECT_THROW(throw rtcd::IoError("x", "f.json", 3, 4), rtcd::Error);
    EXPECT_THROW(throw rtcd::CapacityError("x", "cap", 9, 8), rtcd::Error);
    EXPECT_THROW(throw rtcd::InvalidArgument("x"), std::runtime_error);
    EXPECT_THROW(RTCD_CUDA_CHECK(cudaErrorInvalidValue), rtcd::Error);
    EXPECT_THROW(RTCD_OPTIX_CHECK(OPTIX_ERROR_INVALID_VALUE), rtcd::Error);
}

TEST(CoreLogger, LevelFilteringAndCustomSink) {
    std::vector<std::pair<rtcd::LogLevel, std::string>> got;
    rtcd::Logger log;
    log.level = rtcd::LogLevel::Warning;
    log.sink  = [&](rtcd::LogLevel l, std::string_view m) { got.emplace_back(l, std::string(m)); };
    log.log(rtcd::LogLevel::Error, "e");
    log.log(rtcd::LogLevel::Warning, "w");
    log.log(rtcd::LogLevel::Info, "i");
    log.log(rtcd::LogLevel::Debug, "d");
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0].second, "e");
    EXPECT_EQ(got[1].second, "w");
    log.level = rtcd::LogLevel::Debug;
    log.log(rtcd::LogLevel::Debug, "d");
    EXPECT_EQ(got.size(), 3u);
}

TEST(CoreLogger, DefaultSinkWritesOneLineToStderr) {
    testing::internal::CaptureStderr();
    rtcd::Logger{}.log(rtcd::LogLevel::Error, "hello from the default sink");
    rtcd::Logger::defaultSink(rtcd::LogLevel::Info, "second line\n");
    const std::string out = testing::internal::GetCapturedStderr();
    EXPECT_EQ(out, "[rtcd][error] hello from the default sink\n[rtcd][info] second line\n");
}
