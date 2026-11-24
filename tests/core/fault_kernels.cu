// Test helpers for tests/core: kernels that fail on purpose (error classification of real CUDA failures).
#include <cuda_runtime.h>

namespace rtcd_test {

    __global__ void writeToAddress(unsigned long long address) { *reinterpret_cast<volatile int*>(address) = 1; }
    __global__ void noop() {}

    // Asynchronous illegal-address fault: returns immediately, the error surfaces at the next synchronization.
    void launchIllegalAddressKernel() { writeToAddress<<<1, 1>>>(0x10ull); }

    // Launch-configuration error (too many threads per block): reported by cudaGetLastError(), not sticky.
    void launchInvalidConfiguration() { noop<<<1, 1 << 20>>>(); }

} // namespace rtcd_test
