#pragma once
// rtcd/device_buffer.h -- RAII owners of device memory and page-locked host memory.
//
// Ownership: a DeviceBuffer exclusively owns one cudaMalloc allocation; move-only. The allocation lives on the
// device that was current when it was allocated (the Context's device: the new stack is single-device per
// Context, and every rtcd object makes its Context's device current before allocating).
//
// Allocation policy of the new stack: device memory is allocated at construction time of the owning object,
// sized by capacities (CheckerConfig), never inside CollisionChecker::check() / Backend::prepare(). Therefore
// the allocating functions here are synchronous (cudaMalloc / cudaFree) and are not meant for the hot path.
//
// Threading: a DeviceBuffer is not internally synchronized. Concurrent const use (ptr(), size()) is fine;
// asynchronous copies follow CUDA stream ordering, the caller orders them.
//
// Errors: CUDA failures throw CudaError (allocation failure names the requested size); out-of-range offsets
// throw InvalidArgument (checked on every call, before CUDA is touched).

#include <rtcd/types.h>

#include <cstddef>
#include <span>
#include <type_traits>

namespace rtcd {

    class DeviceBuffer {
    public:
        DeviceBuffer() noexcept = default;
        explicit DeviceBuffer(std::size_t bytes); // allocates; contents uninitialized
        ~DeviceBuffer();                          // cudaFree; never throws (errors go to the default logger)

        DeviceBuffer(DeviceBuffer&& other) noexcept;
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept; // releases the current allocation first
        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DevicePtr ptr() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return bytes_; } // bytes
        bool empty() const noexcept { return bytes_ == 0; }

        // (Re)allocate exactly `bytes` (0 = release). Previous contents are discarded. Synchronous.
        void allocate(std::size_t bytes);
        // Grow-only: reallocates (discarding contents) only if size() < bytes. Synchronous.
        void reserve(std::size_t bytes);
        void release() noexcept;

        // Synchronous copies (cudaMemcpy): return when the copy is complete.
        void upload(const void* src, std::size_t bytes, std::size_t dstOffset = 0);
        void download(void* dst, std::size_t bytes, std::size_t srcOffset = 0) const;

        // Stream-ordered copies / fill (cudaMemcpyAsync / cudaMemsetAsync). For pageable host memory CUDA copies
        // synchronously with respect to the host buffer; for PinnedHostBuffer memory the host buffer must stay
        // valid and unmodified until the stream reaches the copy.
        void uploadAsync(const void* src, std::size_t bytes, StreamHandle stream, std::size_t dstOffset = 0);
        void downloadAsync(void* dst, std::size_t bytes, StreamHandle stream, std::size_t srcOffset = 0) const;
        void copyFromAsync(DevicePtr src, std::size_t bytes, StreamHandle stream, std::size_t dstOffset = 0);
        void memsetAsync(int byteValue, std::size_t bytes, StreamHandle stream, std::size_t dstOffset = 0);

        // Typed helpers (inline; no CUDA needed here).
        template <typename T>
        DeviceSpan<T> span() const noexcept {
            return DeviceSpan<T>{ptr_, bytes_ / sizeof(T)};
        }
        template <typename T>
        static DeviceBuffer fromHost(std::span<const T> data) {
            static_assert(std::is_trivially_copyable_v<T>, "DeviceBuffer::fromHost needs trivially copyable T");
            DeviceBuffer b(data.size_bytes());
            if (!data.empty()) {
                b.upload(data.data(), data.size_bytes());
            }
            return b;
        }

    private:
        DevicePtr ptr_     = 0;
        std::size_t bytes_ = 0;
    };

    // Page-locked host memory (cudaMallocHost), for asynchronous device -> host copies of results and counters
    // that must not stall the copy engine. Move-only; same allocation policy as DeviceBuffer.
    class PinnedHostBuffer {
    public:
        PinnedHostBuffer() noexcept = default;
        explicit PinnedHostBuffer(std::size_t bytes);
        ~PinnedHostBuffer(); // never throws

        PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
        PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;
        PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
        PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

        void* data() noexcept { return ptr_; }
        const void* data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return bytes_; }

        void allocate(std::size_t bytes); // discards contents
        void release() noexcept;

        template <typename T>
        std::span<T> as() noexcept {
            return std::span<T>(static_cast<T*>(ptr_), bytes_ / sizeof(T));
        }

    private:
        void* ptr_         = nullptr;
        std::size_t bytes_ = 0;
    };

} // namespace rtcd
