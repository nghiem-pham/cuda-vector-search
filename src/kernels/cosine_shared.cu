#pragma once
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Kernel v2: shared
// Each block loads a tile of the query vector into shared memory.
// All threads in the block reuse the same tile, reducing global memory reads
// from D reads per thread to D/TILE_SIZE loads per block.
// ─────────────────────────────────────────────

#define TILE_SIZE 32

__global__ void cosine_shared(const float* __restrict__ db,
                               const float* __restrict__ query,
                               float* __restrict__ scores,
                               int N, int D) {
    // Shared memory tile for query vector
    // All threads in a block share this, reducing global memory reads
    __shared__ float s_query[TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // database vector index
    int tid = threadIdx.x;                              // thread index within block

    float dot = 0.0f;

    // Iterate over query vector in tiles of TILE_SIZE
    for (int tile = 0; tile < D; tile += TILE_SIZE) {

        // Each thread loads one element of the query tile into shared memory
        if (tile + tid < D) {
            s_query[tid] = query[tile + tid];
        }
        __syncthreads();  // wait for all threads to finish loading

        // Each thread computes partial dot product for its database vector
        if (idx < N) {
            int end = min(TILE_SIZE, D - tile);
            for (int d = 0; d < end; d++) {
                dot += db[idx * D + (tile + d)] * s_query[d];
            }
        }
        __syncthreads();  // wait before loading next tile
    }

    // Write final similarity score
    if (idx < N) {
        scores[idx] = dot;
    }
}
