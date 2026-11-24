// rtcd_exp_transfer_probe -- Track G2: host cost of moving a few hundred bytes between host and device on this
// machine, by method (the checkHost upload / download of a small check). Times, per method, the host wall time
// from the first call to "data usable" (stream synchronized / copy returned), medians and p10 / p90.
//   H2D: cudaMemcpy from pageable; cudaMemcpy from pinned; memcpy into pinned + cudaMemcpyAsync + sync;
//        a kernel reading the mapped pinned buffer (zero-copy) into device memory + sync.
//   D2H: cudaMemcpy to pinned (the current download); cudaMemcpyAsync to pinned + sync; a kernel writing the
//        device counters into the mapped pinned buffer + sync; baseline: empty kernel + sync, sync alone.
// Also: a check-like sequence "H2D pageable cudaMemcpy, then a kernel on a blocking stream + sync" vs the same
// with the H2D done as an async copy on that stream from pinned memory (is the next work delayed by the DMA?).
// Usage: rtcd_exp_transfer_probe [--bytes 64] [--reps 3000] [--blocked 0|1]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
    __global__ void copyKernel(const unsigned int* src, unsigned int* dst, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = src[i];
    }
    __global__ void emptyKernel() {}

    void ok(cudaError_t e, const char* w) {
        if (e != cudaSuccess) {
            std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
            std::exit(1);
        }
    }
    double pct(std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v[std::size_t(q * double(v.size() - 1))];
    }
} // namespace

int main(int argc, char** argv) {
    std::size_t bytes = 64;
    int reps          = 3000;
    bool blocked      = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--bytes") bytes = std::stoul(argv[i + 1]);
        if (std::string(argv[i]) == "--reps") reps = std::stoi(argv[i + 1]);
        if (std::string(argv[i]) == "--blocked") blocked = std::string(argv[i + 1]) != "0";
    }
    const int n = int(bytes / 4);
    ok(cudaSetDevice(0), "setDevice");
    cudaStream_t s;
    ok(cudaStreamCreate(&s), "stream"); // blocking stream, as the Context's pool streams
    unsigned int *d0, *d1, *pinned;
    std::vector<unsigned int> pageable(n, 1);
    ok(cudaMalloc(&d0, bytes), "malloc");
    ok(cudaMalloc(&d1, bytes), "malloc");
    ok(cudaMallocHost(&pinned, bytes), "mallocHost");
    unsigned int* pinnedDev = nullptr;
    ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pinnedDev), pinned, 0), "hostGetDevicePointer");
    const int blocks = (n + 127) / 128;

    struct M {
        const char* name;
        std::function<void()> f;
        std::vector<double> t;
    };
    std::vector<M> ms{
        {"sync alone (idle stream)", [&] { ok(cudaStreamSynchronize(s), "sync"); }},
        {"empty kernel + sync", [&] { emptyKernel<<<1, 1, 0, s>>>(); ok(cudaStreamSynchronize(s), "sync"); }},
        {"D2D cudaMemcpyAsync + sync", [&] { ok(cudaMemcpyAsync(d1, d0, bytes, cudaMemcpyDeviceToDevice, s), "c"); ok(cudaStreamSynchronize(s), "sync"); }},
        {"H2D cudaMemcpy pageable", [&] { ok(cudaMemcpy(d0, pageable.data(), bytes, cudaMemcpyHostToDevice), "c"); }},
        {"H2D cudaMemcpy pageable + kernel + sync", [&] {
             ok(cudaMemcpy(d0, pageable.data(), bytes, cudaMemcpyHostToDevice), "c");
             copyKernel<<<blocks, 128, 0, s>>>(d0, d1, n);
             ok(cudaStreamSynchronize(s), "sync");
         }},
        {"H2D cudaMemcpy pinned", [&] { ok(cudaMemcpy(d0, pinned, bytes, cudaMemcpyHostToDevice), "c"); }},
        {"H2D memcpy->pinned, async on s + kernel + sync", [&] {
             std::memcpy(pinned, pageable.data(), bytes);
             ok(cudaMemcpyAsync(d0, pinned, bytes, cudaMemcpyHostToDevice, s), "c");
             copyKernel<<<blocks, 128, 0, s>>>(d0, d1, n);
             ok(cudaStreamSynchronize(s), "sync");
         }},
        {"H2D zero-copy kernel (reads mapped pinned) + sync", [&] {
             std::memcpy(pinned, pageable.data(), bytes);
             copyKernel<<<blocks, 128, 0, s>>>(pinnedDev, d1, n);
             ok(cudaStreamSynchronize(s), "sync");
         }},
        {"D2H cudaMemcpy to pinned (current download)", [&] { ok(cudaMemcpy(pinned, d0, bytes, cudaMemcpyDeviceToHost), "c"); }},
        {"D2H cudaMemcpyAsync to pinned + sync", [&] { ok(cudaMemcpyAsync(pinned, d0, bytes, cudaMemcpyDeviceToHost, s), "c"); ok(cudaStreamSynchronize(s), "sync"); }},
        {"D2H zero-copy kernel (writes mapped pinned) + sync", [&] { copyKernel<<<blocks, 128, 0, s>>>(d0, pinnedDev, n); ok(cudaStreamSynchronize(s), "sync"); }},
    };
    if (!blocked) {
        // rotated: every method follows every other one equally often
        for (int r = 0; r < 200 + reps; ++r) {
            for (std::size_t k = 0; k < ms.size(); ++k) {
                M& m = ms[(k + std::size_t(r)) % ms.size()];
                const auto t0 = std::chrono::steady_clock::now();
                m.f();
                const auto t1 = std::chrono::steady_clock::now();
                if (r >= 200) m.t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
    } else {
        // blocked: all repetitions of one method back to back, each preceded by a full device sync
        for (M& m : ms) {
            for (int r = 0; r < 200 + reps; ++r) {
                ok(cudaDeviceSynchronize(), "deviceSync");
                const auto t0 = std::chrono::steady_clock::now();
                m.f();
                const auto t1 = std::chrono::steady_clock::now();
                if (r >= 200) m.t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
    }
    std::printf("%zu bytes, %d reps, median [p10, p90] us\n", bytes, reps);
    for (const M& m : ms) {
        std::printf("  %-52s %8.2f [%7.2f, %7.2f]\n", m.name, pct(m.t, 0.5), pct(m.t, 0.1), pct(m.t, 0.9));
    }
    return 0;
}
