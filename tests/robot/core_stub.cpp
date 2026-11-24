// TEST-LOCAL STUB of rtcd_core (Wave 1 a) -- DELETE AT THE WAVE-1 MERGE.
// Minimal DeviceBuffer and throw helpers so the rtcd_robot GPU tests link before 1a's implementation lands.
// tests/robot/CMakeLists.txt adds this file only while src/core/placeholder.cu still exists (1a deletes the
// placeholder with its first real source), so the stub disappears automatically once rtcd_core is real.
#include <rtcd/device_buffer.h>
#include <rtcd/errors.h>

#include <cuda_runtime.h>

#include <string>

namespace rtcd {

    namespace detail {
        void throwCudaError(int code, const char* expr, const char* file, int line) {
            throw CudaError(std::string("[core stub] ") + cudaGetErrorName(static_cast<cudaError_t>(code)) + " in "
                                + expr + " at " + file + ":" + std::to_string(line),
                code, false, false);
        }
        void throwCuError(int code, const char* expr, const char* file, int line) {
            throw CudaError(std::string("[core stub] CUresult ") + std::to_string(code) + " in " + expr + " at " + file
                                + ":" + std::to_string(line),
                code, true, false);
        }
        void throwOptixError(int code, const char* expr, const char* file, int line, const char*) {
            throw OptixError(std::string("[core stub] OptixResult ") + std::to_string(code) + " in " + expr + " at "
                                 + file + ":" + std::to_string(line),
                code);
        }
    } // namespace detail

    DeviceBuffer::DeviceBuffer(std::size_t bytes) { allocate(bytes); }
    DeviceBuffer::~DeviceBuffer() { release(); }
    DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_   = 0;
        o.bytes_ = 0;
    }
    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_     = o.ptr_;
            bytes_   = o.bytes_;
            o.ptr_   = 0;
            o.bytes_ = 0;
        }
        return *this;
    }
    void DeviceBuffer::allocate(std::size_t bytes) {
        release();
        if (bytes == 0) {
            return;
        }
        void* p = nullptr;
        RTCD_CUDA_CHECK(cudaMalloc(&p, bytes));
        ptr_   = reinterpret_cast<DevicePtr>(p);
        bytes_ = bytes;
    }
    void DeviceBuffer::release() noexcept {
        if (ptr_ != 0) {
            cudaFree(reinterpret_cast<void*>(ptr_));
        }
        ptr_   = 0;
        bytes_ = 0;
    }
    void DeviceBuffer::upload(const void* src, std::size_t bytes, std::size_t dstOffset) {
        if (dstOffset + bytes > bytes_) {
            throw InvalidArgument("[core stub] DeviceBuffer::upload out of range");
        }
        RTCD_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ptr_ + dstOffset), src, bytes, cudaMemcpyHostToDevice));
    }
    void DeviceBuffer::download(void* dst, std::size_t bytes, std::size_t srcOffset) const {
        if (srcOffset + bytes > bytes_) {
            throw InvalidArgument("[core stub] DeviceBuffer::download out of range");
        }
        RTCD_CUDA_CHECK(cudaMemcpy(dst, reinterpret_cast<const void*>(ptr_ + srcOffset), bytes, cudaMemcpyDeviceToHost));
    }
    void DeviceBuffer::memsetAsync(int byteValue, std::size_t bytes, StreamHandle stream, std::size_t dstOffset) {
        if (dstOffset + bytes > bytes_) {
            throw InvalidArgument("[core stub] DeviceBuffer::memsetAsync out of range");
        }
        RTCD_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(ptr_ + dstOffset), byteValue, bytes, stream));
    }

} // namespace rtcd
