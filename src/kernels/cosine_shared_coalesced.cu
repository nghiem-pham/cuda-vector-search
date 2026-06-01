#pragma once
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Kernel v3: shared + coalesced
// Same as cosine_shared but uses column-major (transposed) database layout.
//
// Layout change: db[D x N] instead of db[N x D]
//   db[d * N + idx] instead of db[idx * D + d]
//
//   Thread 0 reads db[d*N + 0]
//   Thread 1 reads db[d*N + 1]   <- consecutive addresses
//   Thread 31 reads db[d*N + 31]
//   -> 1 memory transaction per warp instead of 32
//   -> reduces global memory transactions by ~32x for db access
//
// Caller must transpose the database before loading onto GPU:
//   db_T[d * N + n] = db[n * D + d]  for all n, d
// ─────────────────────────────────────────────

#define TILE_SIZE_COALESCED 32

__global__ void cosine_shared_coalesced(const float* __restrict__ db_T,
                                         const float* __restrict__ query,
                                         float* __restrict__ scores,
                                         int N, int D) {
    // Shared memory tile for query vector
    __shared__ float s_query[TILE_SIZE_COALESCED];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // database vector index
    int tid = threadIdx.x;                              // thread index within block

    float dot = 0.0f;

    // Iterate over query vector in tiles
    for (int tile = 0; tile < D; tile += TILE_SIZE_COALESCED) {

        // Each thread loads one element of the query tile into shared memory
        if (tile + tid < D) {
            s_query[tid] = query[tile + tid];
        }
        __syncthreads();  // wait for all threads to finish loading

        // Each thread computes partial dot product for its database vector
        // db_T is stored as [D x N] -- access db_T[d * N + idx] is coalesced:
        // consecutive threads read consecutive memory addresses
        if (idx < N) {
            int end = min(TILE_SIZE_COALESCED, D - tile);
            #pragma unroll
            for (int d = 0; d < end; d++) {
                dot += db_T[(tile + d) * N + idx] * s_query[d];
            }
        }
        __syncthreads();  // wait before loading next tile
    }

    // Write final similarity score
    if (idx < N) {
        scores[idx] = dot;
    }
}
