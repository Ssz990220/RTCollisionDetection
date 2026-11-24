// rtcd_core: DeviceBuffer, PinnedHostBuffer (rtcd/device_buffer.h).
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include "core_internal.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>

namespace rtcd {

    namespace {
        void* asPtr(DevicePtr p) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(p)); }

        // offset + bytes <= size, without overflow. Throws InvalidArgument naming the operation.
        void checkRange(const char* op, std::size_t offset, std::size_t bytes, std::size_t size) {
            if (offset > size || bytes > size - offset) {
                throw InvalidArgument(std::string("DeviceBuffer::") + op + ": range [" + std::to_string(offset) + ", "
                                      + std::to_string(offset) + " + " + std::to_string(bytes)
                                      + ") exceeds the buffer size of " + std::to_string(size) + " bytes");
            }
        }

        void checkHostPtr(const char* op, const void* p, std::size_t bytes) {
            if (p == nullptr && bytes != 0) {
                throw InvalidArgument(std::string("DeviceBuffer::") + op + ": null host pointer for "
                                      + std::to_string(bytes) + " bytes");
            }
        }
    } // namespace

    // ---- DeviceBuffer ------------------------------------------------------------------------------------------

    DeviceBuffer::DeviceBuffer(std::size_t bytes) { allocate(bytes); }

    DeviceBuffer::~DeviceBuffer() { release(); }

    DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, 0)), bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_   = std::exchange(other.ptr_, 0);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void DeviceBuffer::allocate(std::size_t bytes) {
        release();
        if (bytes == 0) {
            return;
        }
        void* p                = nullptr;
        const cudaError_t code = cudaMalloc(&p, bytes);
        if (code != cudaSuccess) {
            detail::throwCudaErrorMessage(
                code, "DeviceBuffer: cudaMalloc of " + std::to_string(bytes) + " bytes failed");
        }
        ptr_   = static_cast<DevicePtr>(reinterpret_cast<std::uintptr_t>(p));
        bytes_ = bytes;
    }

    void DeviceBuffer::reserve(std::size_t bytes) {
        if (bytes_ < bytes) {
            allocate(bytes);
        }
    }

    void DeviceBuffer::release() noexcept {
        if (ptr_ != 0) {
            const cudaError_t code = cudaFree(asPtr(ptr_));
            if (code != cudaSuccess) {
                detail::logReleaseError("DeviceBuffer: cudaFree", code, false);
            }
        }
        ptr_   = 0;
        bytes_ = 0;
    }

    void DeviceBuffer::upload(const void* src, std::size_t bytes, std::size_t dstOffset) {
        checkRange("upload", dstOffset, bytes, bytes_);
        checkHostPtr("upload", src, bytes);
        if (bytes == 0) {
            return;
        }
        RTCD_CUDA_CHECK(cudaMemcpy(asPtr(ptr_ + dstOffset), src, bytes, cudaMemcpyHostToDevice));
    }

    void DeviceBuffer::download(void* dst, std::size_t bytes, std::size_t srcOffset) const {
        checkRange("download", srcOffset, bytes, bytes_);
        checkHostPtr("download", dst, bytes);
        if (bytes == 0) {
            return;
        }
        RTCD_CUDA_CHECK(cudaMemcpy(dst, asPtr(ptr_ + srcOffset), bytes, cudaMemcpyDeviceToHost));
    }

    void DeviceBuffer::uploadAsync(const void* src, std::size_t bytes, StreamHandle stream, std::size_t dstOffset) {
        checkRange("uploadAsync", dstOffset, bytes, bytes_);
        checkHostPtr("uploadAsync", src, bytes);
        if (bytes == 0) {
            return;
        }
        RTCD_CUDA_CHECK(cudaMemcpyAsync(asPtr(ptr_ + dstOffset), src, bytes, cudaMemcpyHostToDevice, stream));
    }

    void DeviceBuffer::downloadAsync(void* dst, std::size_t bytes, StreamHandle stream, std::size_t srcOffset) const {
        checkRange("downloadAsync", srcOffset, bytes, bytes_);
        checkHostPtr("downloadAsync", dst, bytes);
        if (bytes == 0) {
            return;
        }
        RTCD_CUDA_CHECK(cudaMemcpyAsync(dst, asPtr(ptr_ + srcOffset), bytes, cudaMemcpyDeviceToHost, stream));
    }

    void DeviceBuffer::copyFromAsync(DevicePtr src, std::size_t bytes, StreamHandle stream, std::size_t dstOffset) {
        checkRange("copyFromAsync", dstOffset, bytes, bytes_);
        if (bytes == 0) {
            return;
        }
        if (src == 0) {
            throw InvalidArgument("DeviceBuffer::copyFromAsync: null source device pointer for "
                                  + std::to_string(bytes) + " bytes");
        }
        RTCD_CUDA_CHECK(cudaMemcpyAsync(asPtr(ptr_ + dstOffset), asPtr(src), bytes, cudaMemcpyDeviceToDevice, stream));
    }

    void DeviceBuffer::memsetAsync(int byteValue, std::size_t bytes, StreamHandle stream, std::size_t dstOffset) {
        checkRange("memsetAsync", dstOffset, bytes, bytes_);
        if (bytes == 0) {
            return;
        }
        RTCD_CUDA_CHECK(cudaMemsetAsync(asPtr(ptr_ + dstOffset), byteValue, bytes, stream));
    }

    // ---- PinnedHostBuffer --------------------------------------------------------------------------------------

    PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes) { allocate(bytes); }

    PinnedHostBuffer::~PinnedHostBuffer() { release(); }

    PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

    PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_   = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void PinnedHostBuffer::allocate(std::size_t bytes) {
        release();
        if (bytes == 0) {
            return;
        }
        void* p                = nullptr;
        const cudaError_t code = cudaMallocHost(&p, bytes);
        if (code != cudaSuccess) {
            detail::throwCudaErrorMessage(
                code, "PinnedHostBuffer: cudaMallocHost of " + std::to_string(bytes) + " bytes failed");
        }
        ptr_   = p;
        bytes_ = bytes;
    }

    void PinnedHostBuffer::release() noexcept {
        if (ptr_ != nullptr) {
            const cudaError_t code = cudaFreeHost(ptr_);
            if (code != cudaSuccess) {
                detail::logReleaseError("PinnedHostBuffer: cudaFreeHost", code, false);
            }
        }
        ptr_   = nullptr;
        bytes_ = 0;
    }

} // namespace rtcd
