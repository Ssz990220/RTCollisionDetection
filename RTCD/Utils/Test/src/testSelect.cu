#include <Utils/CUDABuffer.h>
#include <Utils/optix7.h>
#include <cub/cub.cuh>
#include <vector>

using namespace RTCD;

int main() {
    int num_items = 1024;
    CUDABuffer dIn;
    CUDABuffer dFlags;
    CUDABuffer dOut;
    CUDABuffer dCount;

    std::vector<OptixInstance> instances(num_items);
    for (auto& instance : instances) {
        instance.instanceId = std::distance(instances.data(), &instance);
    }

    dIn.alloc_and_upload(instances);
    std::vector<int> flags(num_items);
    // randomly set flags
    for (auto& flag : flags) {
        flag = rand() % 2;
    }
    dFlags.alloc_and_upload(flags);

    dOut.alloc(instances.size() * sizeof(OptixInstance));
    dCount.alloc(sizeof(int));

    // Determine temporary device storage requirements
    CUDABuffer tmp_storage;
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, reinterpret_cast<OptixInstance*>(dIn.d_pointer()),
        reinterpret_cast<int*>(dFlags.d_pointer()), reinterpret_cast<OptixInstance*>(dOut.d_pointer()),
        reinterpret_cast<int*>(dCount.d_pointer()), num_items);

    // Allocate temporary storage
    tmp_storage.alloc(temp_storage_bytes);

    // Run selection
    cub::DeviceSelect::Flagged(reinterpret_cast<void*>(tmp_storage.d_pointer()), temp_storage_bytes,
        reinterpret_cast<OptixInstance*>(dIn.d_pointer()), reinterpret_cast<int*>(dFlags.d_pointer()),
        reinterpret_cast<OptixInstance*>(dOut.d_pointer()), reinterpret_cast<int*>(dCount.d_pointer()), num_items);

    std::vector<OptixInstance> out(num_items);
    dOut.download(out);
    int count;
    dCount.download(&count, 1);

    CUDA_SYNC_CHECK();
}
