#pragma once
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Kernel v1: global
// Each thread computes cosine similarity for one database vector.
// Assumes query and db vectors are L2-normalized -> dot product = cosine.
// No shared memory -- each thread reads query directly from global memory.
// ─────────────────────────────────────────────

__global__ void cosine_global(const float* __restrict__ db,
                               const float* __restrict__ query,
                               float* __restrict__ scores,
                               int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float dot = 0.0f;
    for (int d = 0; d < D; d++) {
        dot += db[idx * D + d] * query[d];
    }
    scores[idx] = dot;
}
