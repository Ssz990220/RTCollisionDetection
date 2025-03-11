template <int degree>
void genCurveOBB(const CUdeviceptr cPts, const CUdeviceptr segs, const CUdeviceptr radii, CUdeviceptr obbs,
    const size_t nSegs, const cudaStream_t stream) {
    const size_t nThreads = 128;
    const size_t nBlocks  = (nSegs + nThreads - 1) / nThreads;
    genCurveOBBKern<degree><<<nBlocks, nThreads, 0, stream>>>(reinterpret_cast<const float3*>(cPts),
        reinterpret_cast<const int*>(segs), reinterpret_cast<const float*>(radii), reinterpret_cast<OBB*>(obbs), nSegs);
}
