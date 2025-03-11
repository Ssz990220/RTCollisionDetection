// dataCpyKern
// @brief Copy data from src to dst based on the flags array
// @tparam flagCondition: condition for copying data, true means copy data for elements whose flags are >0, false
// means
// @param dst: destination pointer
// @param src: source pointer
// @param flags[N]: boolean array indicating whether the data should be copied
// @param eleSize[N]: size of each element, in bytes
// @param whereInDst[N]: offset in dst for each element, in cnt
// @param nEle: number of elements
template <bool flagCondition>
__global__ void dataCpyKern(
    void* dst, const void* src, const uint* flags, const uint* whereInDst, const uint eleSize, const uint nEle) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nEle) {
        if ((flagCondition && flags[idx] > 0) || (!flagCondition && flags[idx] == 0)) {
            // memcpy(dst + whereInDst[idx] * eleSize, src + idx * eleSize, eleSize);
            memcpy(static_cast<char*>(dst) + whereInDst[idx] * eleSize, static_cast<const char*>(src) + idx * eleSize,
                eleSize);
        }
    }
}

template <bool flagCondition>
void compactCpy(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags, const CUdeviceptr whereInDst,
    const uint eleSize, const uint nEle, const cudaStream_t stream) {
    dataCpyKern<flagCondition><<<(nEle + 255) / 256, 256, 0, stream>>>(reinterpret_cast<void*>(dst),
        reinterpret_cast<void*>(src), reinterpret_cast<uint*>(flags), reinterpret_cast<uint*>(whereInDst), eleSize,
        nEle);
}


// dataCpyKern
// @brief Copy data from src to dst based on the flags array
// @tparam flagCondition: condition for copying data, true means copy data for elements whose flags are >0, false
// means
// @param dst: destination pointer
// @param src: source pointer
// @param flags[N]: boolean array indicating whether the data should be copied
// @param eleSize[N]: size of each element, in bytes
// @param whereInDst[N]: offset in dst for each element, in cnt
// @param whereInSrc[N]: offset in src for each element, in cnt
// @param nEle: number of elements
template <bool flagCondition>
__global__ void dataCpyCurveKern(void* dst, const void* src, const uint* flags, const uint* whereInDst,
    const uint* whereInSrc, const uint dstEleSize, const uint srcEleSize, const uint nEle) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nEle) {
        if ((flagCondition && flags[idx] > 0) || (!flagCondition && flags[idx] == 0)) {
            // memcpy(dst + whereInDst[idx] * eleSize, src + whereInSrc[idx] * eleSize, eleCnt[idx] * eleSize);
            memcpy(static_cast<char*>(dst) + whereInDst[idx] * dstEleSize,
                static_cast<const char*>(src) + whereInSrc[idx] * srcEleSize, dstEleSize);
        }
    }
}

template <bool flagCondition>
void compactCpyCurve(CUdeviceptr dst, const CUdeviceptr src, const CUdeviceptr flags, const CUdeviceptr whereInDst,
    const CUdeviceptr whereInSrc, const uint dstEleSize, const uint srcEleSize, const uint nEle,
    const cudaStream_t stream) {
    dataCpyCurveKern<flagCondition><<<(nEle + 255) / 256, 256, 0, stream>>>(reinterpret_cast<void*>(dst),
        reinterpret_cast<void*>(src), reinterpret_cast<uint*>(flags), reinterpret_cast<uint*>(whereInDst),
        reinterpret_cast<uint*>(whereInSrc), dstEleSize, srcEleSize, nEle);
}

// streamCompact
// @brief Compact the stream based on the isValid array. The element has constant size thus no `whereInSrc`,
// `streamEleCnt` is needed. The size of each element is specified by `eleSize`.
template <bool flagCondition>
void streamCompact(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer, uint* nValid,
    const uint eleSize, const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaMemcpyKind kind,
    const cudaStream_t stream) {
    transformArray<uint>(flags, tempBuffer, is_valid<flagCondition>(), nEle, stream);
    // scanExclusiveShort(tempBuffer, whereInDst, nEle, stream);
    exclusiveScan(whereInDst, tempBuffer, tempBuffer + nEle * eleSize, nEle, stream);
    compactCpy<flagCondition>(dst, src, flags, whereInDst, eleSize, nEle, stream);

    reduceSum<uint>(tempBuffer, tempBuffer, nEle, stream);
    cudaMemcpyAsync(nValid, reinterpret_cast<void*>(tempBuffer), sizeof(uint), kind, stream);
}

template <bool flagCondition>
void streamCompact(const CUdeviceptr src, const CUdeviceptr flags, CUdeviceptr tempBuffer, const uint eleSize,
    const uint nEle, CUdeviceptr dst, CUdeviceptr whereInDst, const cudaStream_t stream) {
    transformArray<uint>(flags, tempBuffer, is_valid<flagCondition>(), nEle, stream);
    // scanExclusiveShort(tempBuffer, whereInDst, nEle, stream);
    exclusiveScan(whereInDst, tempBuffer, tempBuffer + nEle * eleSize, nEle, stream);
    compactCpy<flagCondition>(dst, src, flags, whereInDst, eleSize, nEle, stream);
}
