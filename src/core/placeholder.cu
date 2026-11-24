// Skeleton placeholder (checks that the new tree compiles CUDA). Delete when the first real source is added.
namespace rtcd::detail {
    __global__ void corePlaceholderKernel(int* p) {
        if (p) {
            *p = 0;
        }
    }
    void corePlaceholder() {}
} // namespace rtcd::detail
