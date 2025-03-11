#include <Utils/CUDABuffer.h>
#include <Utils/cuda/cudaAlgorithms.cuh>
#include <iostream>
#include <numeric>
#include <vector>

using namespace RTCD;

CUDABuffer tempBuffer;

template <typename T>
void testType(std::vector<uint>& nEle) {
    for (auto numElements : nEle) {
        std::vector<T> input(numElements);
        std::iota(input.begin(), input.end(), 1);


        CUDABuffer d_input;
        d_input.alloc_and_upload(input);

        reduceSum<T>(d_input.d_pointer(), tempBuffer.d_pointer(), numElements, 0);
        CUDA_SYNC_CHECK();
        // std::vector<int> output(input.size());
        T output;
        CUDA_CHECK(
            cudaMemcpy(&output, reinterpret_cast<void*>(tempBuffer.d_pointer()), sizeof(T), cudaMemcpyDeviceToHost));

        T expectedOutput = std::accumulate(input.begin(), input.end(), 0);

        if (output == expectedOutput) {
            std::cout << "Test passed for " << numElements << " elements" << std::endl;
        } else {
            std::cout << "Test failed for " << numElements << " elements" << std::endl;
            std::cout << "Expected: " << expectedOutput << " Got: " << output << std::endl;
            throw std::runtime_error("Test failed");
        }
    }
}

int main() {
    std::vector<uint> nEle = {4, 5, 8, 1023, 1024, 2047, 2048, 2049, 4096};
    tempBuffer.alloc(4096 * sizeof(size_t));

    std::cout << "Testing for uint" << std::endl;
    testType<uint>(nEle);
    std::cout << "Testing for int" << std::endl;
    testType<int>(nEle);
    std::cout << "Testing for float" << std::endl;
    testType<float>(nEle);
    std::cout << "Testing for size_t" << std::endl;
    testType<size_t>(nEle);
}
