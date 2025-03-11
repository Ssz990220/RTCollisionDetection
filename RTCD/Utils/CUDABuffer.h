#pragma once
// common std stuff
#include "CUDAMacro.h"
#include <array>
#include <assert.h>
#include <cuda_runtime.h>
#include <vector>

namespace RTCD {

    /*! simple wrapper for creating, and managing a device-side CUDA
        buffer */
    struct CUDABuffer {
    public: // Constructors & destructor
        size_t elsize;

        CUDABuffer() {
            sizeInBytes = 0;
            d_ptr       = nullptr;
            elsize      = 0;
        }

        template <typename T>
        CUDABuffer(const std::vector<T>& vt) {
            alloc_and_upload(vt);
        }

        CUDABuffer(size_t sizeInBytes) { alloc(sizeInBytes); }

        CUDABuffer(CUDABuffer&& other) noexcept { swap(other); }

        CUDABuffer& operator=(CUDABuffer&& other) noexcept {
            free();
            swap(other);
            return *this;
        }

        ~CUDABuffer() { free(); }

        void swap(CUDABuffer& other) {
            using std::swap;
            swap(sizeInBytes, other.sizeInBytes);
            swap(d_ptr, other.d_ptr);
        }

        size_t sizeInBytes{0};

    protected: // no copy constructor allowed, only move
        CUDABuffer(const CUDABuffer& other)            = delete;
        CUDABuffer& operator=(const CUDABuffer& other) = delete;

    public:
        inline CUdeviceptr offset_d_pointer(size_t offset) const {
            return (CUdeviceptr) d_ptr + elsize * offset + elsize * (offset == 0);
        }

        inline CUdeviceptr d_pointer() const { return (CUdeviceptr) d_ptr; }

        void copy(CUDABuffer& other) {
            assert(sizeInBytes == other.sizeInBytes);
            CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, sizeInBytes, cudaMemcpyDeviceToDevice));
        }

        void copy(CUdeviceptr other, size_t size) {
            assert(sizeInBytes == size);
            assert(d_ptr != nullptr);
            CUDA_CHECK(cudaMemcpy(d_ptr, (void*) other, sizeInBytes, cudaMemcpyDeviceToDevice));
        }

        void move(CUDABuffer& other) {
            assert(d_ptr == nullptr);
            d_ptr       = other.d_ptr;
            sizeInBytes = other.sizeInBytes;
        }

        //! re-size buffer to given number of bytes
        void resize(size_t size) {
            if (d_ptr) {
                free();
            }
            alloc(size);
        }

        void expandIfNotEnough(size_t size) {
            if (sizeInBytes < size) {
                free();
                alloc(size);
            }
        }

        void resizeAsync(size_t size, cudaStream_t& stream) {
            if (d_ptr) {
                freeAsync(stream);
            }
            allocAsync(size, stream);
        }

        //! allocate to given number of bytes
        void alloc(size_t size) {
            assert(d_ptr == nullptr);
            this->sizeInBytes = size;
            CUDA_CHECK(cudaMalloc((void**) &d_ptr, sizeInBytes));
        }

        template <typename T>
        void allocManaged(size_t sizeInT) {
            assert(d_ptr == nullptr);
            this->sizeInBytes = sizeInT * sizeof(T);
            CUDA_CHECK(cudaMallocManaged((void**) &d_ptr, sizeInBytes));
        }

        void allocAsync(size_t size, cudaStream_t& stream) {
            assert(d_ptr == nullptr);
            this->sizeInBytes = size;
            CUDA_CHECK(cudaMallocAsync((void**) &d_ptr, sizeInBytes, stream));
        }

        //! free allocated memory
        void free() {
            if (d_ptr == nullptr) {
                return;
            }
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr       = nullptr;
            sizeInBytes = 0;
        }

        void freeAsync(cudaStream_t stream) {
            CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
            d_ptr       = nullptr;
            sizeInBytes = 0;
        }

    public:
        // Upload & Download
        template <typename T>
        void alloc_and_upload(const std::vector<T>& vt) {
            alloc(vt.size() * sizeof(T));
            upload((const T*) vt.data(), vt.size());
            elsize = sizeof(T);
        }

        template <typename T, size_t N>
        void alloc_and_upload(const std::array<T, N>& at) {
            alloc(at.size() * sizeof(T));
            upload((const T*) at.data(), at.size());
            elsize = sizeof(T);
        }

        template <typename T>
        void alloc_and_upload(const T* t, size_t count) {
            alloc(count * sizeof(T));
            upload(t, count);
            elsize = sizeof(T);
        }

        template <typename T>
        void upload(const std::vector<T>& t) {
            assert(d_ptr != nullptr);
            assert(sizeInBytes >= t.size() * sizeof(T));
            CUDA_CHECK(cudaMemcpy(d_ptr, (void*) t.data(), t.size() * sizeof(T), cudaMemcpyHostToDevice));
            elsize = sizeof(T);
        }

        template <typename T>
        void upload(const T* t, size_t count) {
            assert(d_ptr != nullptr);
            // assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(cudaMemcpy(d_ptr, (void*) t, count * sizeof(T), cudaMemcpyHostToDevice));
            elsize = sizeof(T);
        }

        template <typename T>
        void uploadAsync(const T* t, size_t count, cudaStream_t& stream) {
            assert(d_ptr != nullptr);
            // assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void*) t, count * sizeof(T), cudaMemcpyHostToDevice, stream));
            elsize = sizeof(T);
        }

        template <typename T>
        void download(T* t, size_t count) {
            assert(d_ptr != nullptr);
            assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(cudaMemcpy((void*) t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        template <typename T>
        void downloadAsync(T* t, size_t count, cudaStream_t& stream) {
            assert(d_ptr != nullptr);
            assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(cudaMemcpyAsync((void*) t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }

        template <typename T>
        void download(std::vector<T>& vt) {
            assert(d_ptr != nullptr);
            vt.resize(sizeInBytes / sizeof(T));
            CUDA_CHECK(cudaMemcpy(vt.data(), d_ptr, vt.size() * sizeof(T), cudaMemcpyDeviceToHost));
        }

        template <typename T>
        void downloadAsync(std::vector<T>& vt, cudaStream_t stream) {
            assert(d_ptr != nullptr);
            vt.resize(sizeInBytes / sizeof(T));
            CUDA_CHECK(cudaMemcpyAsync(vt.data(), d_ptr, vt.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }


        template <typename T>
        void downloadPartial(T* t, size_t count) {
            assert(d_ptr != nullptr);
            assert(sizeInBytes > count * sizeof(T));
            CUDA_CHECK(cudaMemcpy((void*) t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }

    private:
        void* d_ptr{nullptr};
    };

    struct CUDAPitchBuffer {
    private:
        void* d_ptr{nullptr};

    public:
        size_t w{0};
        size_t h{0};
        size_t pitch{0};

    public:
        CUDAPitchBuffer() { d_ptr = nullptr; }

        CUDAPitchBuffer(const size_t widthInBytes, const size_t height) {
            alloc(widthInBytes, height);
            w = widthInBytes;
            h = height;
        }

        CUDAPitchBuffer(CUDABuffer& other, const size_t widthInBytes, const size_t height) {
            fromCUDABuffer(other, widthInBytes, height);
        }

        template <typename T>
        CUDAPitchBuffer(T* t, const size_t widthInBytes, const size_t height) {
            alloc(widthInBytes, height);
            CUDA_CHECK(cudaMemcpy2D(d_ptr, pitch, t, widthInBytes, widthInBytes, height, cudaMemcpyHostToDevice));
        }

        ~CUDAPitchBuffer() { free(); }

        // Move constructor
        CUDAPitchBuffer(CUDAPitchBuffer&& other) noexcept { swap(other); }

        CUDAPitchBuffer& operator=(CUDAPitchBuffer&& other) noexcept {
            free();
            swap(other);
            return *this;
        }

        inline CUdeviceptr d_pointer() const { return (CUdeviceptr) d_ptr; }


        void alloc(const size_t widthInBytes, const size_t height) {
            assert(d_ptr == nullptr);
            w = widthInBytes;
            h = height;
            CUDA_CHECK(cudaMallocPitch((void**) &d_ptr, &pitch, widthInBytes, height));
        }

        void free() {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
            pitch = 0;
            w     = 0;
            h     = 0;
        }

        template <typename T>
        void upload(T* t, cudaMemcpyKind kind) {
            CUDA_CHECK(cudaMemcpy2D(d_ptr, pitch, t, w, w, h, kind));
        }

        template <typename T>
        void uploadAsync(T* t, cudaStream_t stream, cudaMemcpyKind kind) {
            CUDA_CHECK(cudaMemcpy2DAsync(d_ptr, pitch, t, w, w, h, kind, stream));
        }


        template <typename T>
        void alloc_and_upload(T* t, const size_t widthInBytes, const size_t height, cudaMemcpyKind kind) {
            alloc(widthInBytes, height);
            upload(t, kind);
        }

        void fromCUDABuffer(CUDABuffer& other, const size_t widthInBytes, const size_t height) {
            alloc(widthInBytes, height);
            CUDA_CHECK(cudaMemcpy2D(
                d_ptr, pitch, (void*) other.d_pointer(), widthInBytes, widthInBytes, height, cudaMemcpyDeviceToDevice));
        }

        void download(void* t, const size_t widthInBytes) {
            CUDA_CHECK(cudaMemcpy2D((void*) t, widthInBytes, d_ptr, pitch, w, h, cudaMemcpyDeviceToHost));
        }

    private:
        void swap(CUDAPitchBuffer& other) {
            using std::swap;
            swap(d_ptr, other.d_ptr);
            swap(pitch, other.pitch);
            swap(w, other.w);
            swap(h, other.h);
        }

    protected:
        CUDAPitchBuffer(const CUDAPitchBuffer& other)            = delete;
        CUDAPitchBuffer& operator=(const CUDAPitchBuffer& other) = delete;
    };

} // namespace RTCD
