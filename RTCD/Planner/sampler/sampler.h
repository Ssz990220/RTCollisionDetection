#pragma once

#include <Planner/constant.h>
#include <Utils/CUDABuffer.h>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


float localHaltonSingleNumber(const int n, const int b) {
    float hn = 0;
    int n0   = n;
    float f  = 1 / ((float) b);

    while (n0 > 0) {
        float n1 = n0 / b;
        int r    = n0 - n1 * b;
        hn += f * r;
        f  = f / b;
        n0 = n1;
    }
    return hn;
}

template <size_t DOF>
void createSamplesHalton(const size_t nSample, int skip, std::vector<float>& samples, std::array<float, DOF>& initial,
    std::array<float, DOF>& goal, std::array<float, DOF>& lo, std::array<float, DOF>& hi) {
    const int numPrimes = 25;
    const int bases[25] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    assert(skip + DOF < numPrimes && "skip in sampling halton seq too high");

    for (int d = 0; d < DOF; ++d) {
        for (int n = 0; n < nSample; ++n) {
            samples[n * DOF + d] = localHaltonSingleNumber(n, bases[d + skip]) * (hi[d] - lo[d]) + lo[d];
        }
    }

    // replace goal and initial nodes
    for (int d = 0; d < DOF; ++d) {
        samples[d]                       = initial[d];
        samples[(nSample - 1) * DOF + d] = goal[d];
    }
}

template <size_t DOF>
void createSamplesHalton(const size_t nSample, std::vector<std::array<float, DOF>>& samples,
    std::array<float, DOF>& initial, std::array<float, DOF>& goal, std::array<float, DOF>& lo,
    std::array<float, DOF>& hi, int skip = 0) {
    const int numPrimes = 25;
    const int bases[25] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    assert(skip + DOF < numPrimes && "skip in sampling halton seq too high");

    for (int d = 0; d < DOF; ++d) {
        for (int n = 0; n < nSample; ++n) {
            samples[n][d] = localHaltonSingleNumber(n, bases[d + skip]) * (hi[d] - lo[d]) + lo[d];
        }
    }

    // replace goal and initial nodes
    for (int d = 0; d < DOF; ++d) {
        samples[0][d]           = initial[d];
        samples[nSample - 1][d] = goal[d];
    }
}

template <size_t DOF>
void createSamplesHalton(const size_t nSample, std::vector<std::array<float, DOF>>& samples, std::array<float, DOF>& lo,
    std::array<float, DOF>& hi, int skip = 0) {
    const int numPrimes = 25;
    const int bases[25] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    assert(skip + DOF < numPrimes && "skip in sampling halton seq too high");

    for (int d = 0; d < DOF; ++d) {
        for (int n = 0; n < nSample; ++n) {
            samples[n][d] = localHaltonSingleNumber(n, bases[d + skip]) * (hi[d] - lo[d]) + lo[d];
        }
    }
}
