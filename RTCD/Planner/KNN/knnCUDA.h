#pragma once
#include <Planner/constant.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


void knn_cuda_global(float* ref, size_t ref_pitch, size_t ref_width, float* knn_dist, size_t dist_pitch, int* knn_index,
    size_t index_pitch, int N, int D, int K, cudaStream_t stream = 0);

void knn_cuda_global_mask(float* ref, size_t ref_pitch, int* mask, float* knn_dist, size_t dist_pitch, int* knn_index,
    size_t index_pitch, int N, int D, int K, cudaStream_t stream = 0);

cudaError_t knn_cuda_set_weight(const float* weight, const size_t d);
