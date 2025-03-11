#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>

__device__ int binary_search(const int* prefix_sum, int size, int id) {
    int left  = 0;
    int right = size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (prefix_sum[mid] <= id && (mid == size - 1 || prefix_sum[mid + 1] > id)) {
            return mid;
        }

        if (prefix_sum[mid] <= id) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1; // If not found, which should not happen with correct input
}

__global__ void find_ray_index(const int* prefix_sum, int size, int id, int* result) {
    *result = binary_search(prefix_sum, size, id);
}

int main() {
    const int h_array[]      = {0, 3, 0, 7, 1};
    const int h_prefix_sum[] = {0, 3, 10, 14, 15};
    int size                 = sizeof(h_prefix_sum) / sizeof(h_prefix_sum[0]);
    int h_result;

    int* d_prefix_sum;
    int* d_result;

    cudaMalloc((void**) &d_prefix_sum, size * sizeof(int));
    cudaMalloc((void**) &d_result, sizeof(int));

    cudaMemcpy(d_prefix_sum, h_prefix_sum, size * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> test_array{0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    for (auto& id : test_array) {

        find_ray_index<<<1, 1>>>(d_prefix_sum, size, id, d_result);

        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        printf("The ray with ID %d belongs to array index %d\n", id, h_result);
    }

    cudaFree(d_prefix_sum);
    cudaFree(d_result);

    return 0;
}
