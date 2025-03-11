#include "knnCUDA.h"

#define BLOCK_DIM 16

__constant__ float knnWeight[8];

__global__ void compute_distances(
    float* ref, int ref_width, int ref_pitch, float* query, int query_width, int query_pitch, int height, float* dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height - 1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 =
        (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a / ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0) ? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1) ? query[b + query_pitch * ty + tx] : 0;
        } else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k) {
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp * tmp * knnWeight[k] * knnWeight[k];
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and
        // B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        if (blockDim.x * blockIdx.x + tx == blockDim.y * blockIdx.y + ty) {
            ssd = LARGEST_FLOAT;
        }
        dist[(begin_A + ty) * query_pitch + begin_B + tx] = ssd;
    }
}

__global__ void compute_distances(float* ref, int ref_width, int* mask, int ref_pitch, float* query, int query_width,
    int query_pitch, int height, float* dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    // shared_A should be of dim: height * BLOCK_DIM
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx    = threadIdx.x;
    int ty    = threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + tx;
    int idx_y = blockDim.y * blockIdx.y + ty;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height - 1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 =
        (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all dimensions in case DIM > BLOCK_DIM
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a / ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0) ? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1) ? query[b + query_pitch * ty + tx] : 0;
        } else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k) {
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp * tmp * knnWeight[k] * knnWeight[k];
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and
        // B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        if (idx_x == idx_y || mask[idx_x] || mask[idx_y]) {
            ssd = LARGEST_FLOAT;
        }
        dist[(begin_A + ty) * query_pitch + begin_B + tx] = ssd;
    }
}

__global__ void modified_insertion_sort(
    float* dist, int dist_pitch, int* index, int index_pitch, int width, int height, int k) {

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float* p_dist = dist + xIndex;
        int* p_index  = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i = 1; i < height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i * dist_pitch];
            int curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k - 1) * dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k - 1);
            while (j > 0 && p_dist[(j - 1) * dist_pitch] > curr_dist) {
                p_dist[j * dist_pitch]   = p_dist[(j - 1) * dist_pitch];
                p_index[j * index_pitch] = p_index[(j - 1) * index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j * dist_pitch]   = curr_dist;
            p_index[j * index_pitch] = curr_index;
        }
    }
}


__global__ void compute_sqrt(float* dist, int width, int pitch, int k) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < width && yIndex < k) {
        dist[yIndex * pitch + xIndex] = sqrt(dist[yIndex * pitch + xIndex]);
    }
}


__global__ void compute_squared_norm(float* array, int width, int pitch, int height, float* norm) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex < width) {
        float sum = 0.f;
        for (int i = 0; i < height; i++) {
            float val = array[i * pitch + xIndex];
            sum += val * val;
        }
        norm[xIndex] = sum;
    }
}

__global__ void add_reference_points_norm(float* array, int width, int pitch, int height, float* norm) {
    unsigned int tx     = threadIdx.x;
    unsigned int ty     = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx == 0 && yIndex < height) {
        shared_vec[ty] = norm[yIndex];
    }
    __syncthreads();
    if (xIndex < width && yIndex < height) {
        array[yIndex * pitch + xIndex] += shared_vec[ty];
    }
}


__global__ void add_query_points_norm_and_sqrt(float* array, int width, int pitch, int k, float* norm) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < width && yIndex < k) {
        array[yIndex * pitch + xIndex] = sqrt(array[yIndex * pitch + xIndex] + norm[xIndex]);
    }
}

void knn_cuda_global(float* ref, size_t ref_pitch, size_t ref_width, float* knn_dist, size_t dist_pitch, int* knn_index,
    size_t index_pitch, int N, int D, int K, cudaStream_t stream) {

    // Compute the squared Euclidean distances
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0((ref_width + BLOCK_DIM - 1) / BLOCK_DIM, (ref_width + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    // compute_distances<<<grid0, block0>>>(ref, ref_pitch, knn_dist);

    compute_distances<<<grid0, block0, 0, stream>>>(ref, ref_width, ref_pitch, ref, ref_width, ref_pitch, D, knn_dist);

    // Sort the distances with their respective indexes
    dim3 block1(128, 1, 1);
    dim3 grid1((ref_width + 127) / 128, 1, 1);
    modified_insertion_sort<<<grid1, block1, 0, stream>>>(
        knn_dist, dist_pitch, knn_index, index_pitch, ref_width, ref_width, K);

    // Compute the square root of the k smallest distances
    dim3 block2(16, 16, 1);
    dim3 grid2((ref_width + 15) / 16, (K + 15) / 16, 1);
    compute_sqrt<<<grid2, block2, 0, stream>>>(knn_dist, ref_width, ref_pitch, K);
}

void knn_cuda_global_mask(float* ref, size_t ref_pitch, int* mask, float* knn_dist, size_t dist_pitch, int* knn_index,
    size_t index_pitch, int N, int D, int K, cudaStream_t stream) {

    // Compute the squared Euclidean distances
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    // compute_distances<<<grid0, block0>>>(ref, ref_pitch, knn_dist);

    compute_distances<<<grid0, block0, 0, stream>>>(ref, N, mask, ref_pitch, ref, N, ref_pitch, D, knn_dist);

    // Sort the distances with their respective indexes
    dim3 block1(128, 1, 1);
    dim3 grid1((N + 127) / 128, 1, 1);
    modified_insertion_sort<<<grid1, block1, 0, stream>>>(knn_dist, dist_pitch, knn_index, index_pitch, N, N, K);

    // Compute the square root of the k smallest distances
    dim3 block2(16, 16, 1);
    dim3 grid2((N + 15) / 16, (K + 15) / 16, 1);
    compute_sqrt<<<grid2, block2, 0, stream>>>(knn_dist, N, ref_pitch, K);
}

cudaError_t knn_cuda_set_weight(const float* weight, const size_t d) {
    return cudaMemcpyToSymbol(knnWeight, weight, d * sizeof(float));
}
