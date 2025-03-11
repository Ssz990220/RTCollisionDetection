#pragma once
#include "knnCUDA.h"
#include <Utils/CUDABuffer.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace RTCD {
    namespace KNN {
        // A GPU implementation of KNN
        template <size_t N, size_t D, size_t K>
        class KNN {
        protected:
            CUDAPitchBuffer dRef;
            CUDAPitchBuffer dDist;
            CUDAPitchBuffer dIdx;
            size_t refWidth;
            size_t refPitch;
            size_t distPitch;
            size_t idxPitch;

        public:
            KNN() {
                refWidth = N;
                dRef.alloc(N * sizeof(float), D);
                dDist.alloc(N * sizeof(float), N);
                dIdx.alloc(N * sizeof(int), K);

                refPitch  = dRef.pitch / sizeof(float);
                distPitch = dDist.pitch / sizeof(float);
                idxPitch  = dIdx.pitch / sizeof(int);

                assert(distPitch == refPitch && idxPitch == refPitch && "Pitch must be the same");
            }

            KNN(const int sizePoints) {
                refWidth = sizePoints;
                dRef.alloc(sizePoints * sizeof(float), D);
                dDist.alloc(sizePoints * sizeof(float), sizePoints);
                dIdx.alloc(sizePoints * sizeof(int), K);

                refPitch  = dRef.pitch / sizeof(float);
                distPitch = dDist.pitch / sizeof(float);
                idxPitch  = dIdx.pitch / sizeof(int);

                assert(distPitch == refPitch && idxPitch == refPitch && "Pitch must be the same");
            }

            KNN(float* ref, cudaMemcpyKind kind) {
                refWidth = N;
                dRef.alloc_and_upload(ref, N * sizeof(float), D, kind);
                dDist.alloc(N * sizeof(float), N);
                dIdx.alloc(N * sizeof(int), K);

                refPitch  = dRef.pitch / sizeof(float);
                distPitch = dDist.pitch / sizeof(float);
                idxPitch  = dIdx.pitch / sizeof(int);

                assert(distPitch == refPitch && idxPitch == refPitch && "Pitch must be the same");
            }

            KNN(CUDABuffer& buffer) {
                refWidth = N;
                dRef.fromCUDABuffer(buffer, N * sizeof(float), D);
                dDist.alloc(N * sizeof(float), N);
                dIdx.alloc(N * sizeof(int), K);

                refPitch  = dRef.pitch / sizeof(float);
                distPitch = dDist.pitch / sizeof(float);
                idxPitch  = dIdx.pitch / sizeof(int);

                assert(distPitch == refPitch && idxPitch == refPitch && "Pitch must be the same");
            }

            void setWeight(const float* weight) {
                CUDA_CHECK(knn_cuda_set_weight(weight, D));
            }

            void setRef(float* ref, cudaMemcpyKind kind) {
                dRef.upload(ref, kind);
            }

            void setRef(float* ref, cudaStream_t stream, cudaMemcpyKind kind) {
                dRef.uploadAsync(ref, stream, kind);
            }

            void NN(cudaStream_t stream = 0) {
                knn_cuda_global((float*) dRef.d_pointer(), dRef.pitch / sizeof(float), refWidth,
                    (float*) dDist.d_pointer(), dDist.pitch / sizeof(float), (int*) dIdx.d_pointer(),
                    dIdx.pitch / sizeof(int), N, D, K);
            }

            void NN(CUDABuffer& mask, cudaStream_t stream = 0) {
                knn_cuda_global_mask((float*) dRef.d_pointer(), dRef.pitch / sizeof(float), (int*) mask.d_pointer(),
                    (float*) dDist.d_pointer(), dDist.pitch / sizeof(float), (int*) dIdx.d_pointer(),
                    dIdx.pitch / sizeof(int), N, D, K);
            }

            void downloadResult(float* dist, int* idx, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
                CUDA_CHECK(cudaMemcpy2D((void*) dist, refWidth * sizeof(float), (void*) dDist.d_pointer(), dDist.pitch,
                    refWidth * sizeof(float), K, kind));
                CUDA_CHECK(cudaMemcpy2D((void*) idx, refWidth * sizeof(int), (void*) dIdx.d_pointer(), dIdx.pitch,
                    refWidth * sizeof(int), K, kind));
            }

            void downloadResult(
                float* dist, int* idx, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
                CUDA_CHECK(cudaMemcpy2DAsync((void*) dist, refWidth * sizeof(float), (void*) dDist.d_pointer(),
                    dDist.pitch, refWidth * sizeof(float), K, kind, stream));
                CUDA_CHECK(cudaMemcpy2DAsync((void*) idx, refWidth * sizeof(int), (void*) dIdx.d_pointer(), dIdx.pitch,
                    refWidth * sizeof(int), K, kind, stream));
            }
        };
    }; // namespace KNN
} // namespace RTCD
