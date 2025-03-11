
template <class T>
__global__ void reduceSumKern(const T* g_idata, T* g_odata, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata             = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < N) ? g_idata[i] : 0;

    if (i + blockDim.x < N) {
        mySum += g_idata[i + blockDim.x];
    }

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

template <class T>
void reduceSum(const CUdeviceptr d_idata, CUdeviceptr d_output, const uint size, const cudaStream_t stream) {
    constexpr uint threads = 512;
    uint reducedSize       = size;
    uint roundedN          = roundUpLog2(reducedSize);
    uint smemSize          = 2 * threads * sizeof(T);
    uint numBlocks         = (roundedN + 512 - 1) / 512;
    uint iter              = 0;
    while (reducedSize > 1) {
        if (iter == 0) {
            reduceSumKern<T><<<numBlocks, 512, smemSize, stream>>>(
                reinterpret_cast<T*>(d_idata), reinterpret_cast<T*>(d_output), reducedSize);
        } else {
            reduceSumKern<T><<<numBlocks, 512, smemSize, stream>>>(
                reinterpret_cast<T*>(d_output), reinterpret_cast<T*>(d_output), reducedSize);
        }
        reducedSize = numBlocks;
        numBlocks   = (numBlocks >>= 10) ? (numBlocks >>= 10) : 1;
        iter++;
    }
}

template <class T>
__global__ void tfReduceSumKern(const T* g_idata, const T* g_weight, T* g_odata, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata             = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < N) ? g_idata[i] * g_weight[i] : 0;

    if (i + blockDim.x < N) {
        mySum += g_idata[i + blockDim.x] * g_weight[i + blockDim.x];
    }

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

template <class T>
void transform_reduce(const CUdeviceptr d_idata, const CUdeviceptr d_weight, CUdeviceptr d_odata, const uint size,
    const cudaStream_t stream) {

    constexpr uint threads = 512;
    uint reducedSize       = size;
    uint roundedN          = roundUpLog2(reducedSize);
    uint smemSize          = 2 * threads * sizeof(T);
    uint numBlocks         = (roundedN + 512 - 1) / 512;
    uint iter              = 0;
    while (reducedSize > 1) {
        if (iter == 0) {
            tfReduceSumKern<T><<<numBlocks, 512, smemSize, stream>>>(reinterpret_cast<T*>(d_idata),
                reinterpret_cast<T*>(d_weight), reinterpret_cast<T*>(d_odata), reducedSize);
        } else {
            reduceSumKern<T><<<numBlocks, 512, smemSize, stream>>>(
                reinterpret_cast<T*>(d_odata), reinterpret_cast<T*>(d_odata), reducedSize);
        }
        reducedSize = numBlocks;
        numBlocks   = (numBlocks >>= 10) ? (numBlocks >>= 10) : 1;
        iter++;
    }
}
