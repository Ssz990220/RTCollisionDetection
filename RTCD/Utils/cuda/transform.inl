
template <typename T, typename Functor>
__global__ void transformKernel(const T* d_input, const T* d_input2, T* d_output, const size_t size, Functor functor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_output[idx] = functor(d_input[idx], d_input2[idx]);
    }
}

template <typename T, typename Functor>
void transformArray(const CUdeviceptr input, const CUdeviceptr input2, CUdeviceptr output, Functor functor,
    const size_t size, const cudaStream_t stream) {
    // Launch the kernel with appropriate block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
    transformKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(input), reinterpret_cast<T*>(input2), reinterpret_cast<T*>(output), size, functor);
}


template <typename T, typename Functor>
__global__ void transformKernel(const T* d_input, T* d_output, const size_t size, Functor functor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_output[idx] = functor(d_input[idx]);
    }
}


template <typename T, typename Functor>
void transformArray(
    const CUdeviceptr input, CUdeviceptr output, Functor functor, const size_t size, const cudaStream_t stream) {
    // Launch the kernel with appropriate block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
    transformKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(input), reinterpret_cast<T*>(output), size, functor);
}

template <typename T, bool flagCondition>
void transformIsValid(const CUdeviceptr input, CUdeviceptr output, const size_t size, const cudaStream_t stream) {
    // Launch the kernel with appropriate block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
    transformKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<T*>(input), reinterpret_cast<T*>(output), size, is_valid<flagCondition>());
}
