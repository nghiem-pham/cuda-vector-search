#include "engine.h"
#include "kernels/cosine_global.cu"
#include "kernels/cosine_shared.cu"
#include "kernels/cosine_shared_coalesced.cu"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(err) + " at " __FILE__ ":"                \
                + std::to_string(__LINE__));                                    \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────
// Engine implementation
// ─────────────────────────────────────────────

Engine::Engine(const float* db, int N, int D)
    : n_(N), d_(D) {
    size_t bytes = (size_t)N * D * sizeof(float);

    // Load original layout
    CUDA_CHECK(cudaMalloc(&d_db_, bytes));
    CUDA_CHECK(cudaMemcpy(d_db_, db, bytes, cudaMemcpyHostToDevice));

    // Transpose on CPU then copy
    std::vector<float> db_T(N*D);
    for (int d = 0; d < D; d++) {
        for (int n = 0; n < N; n++) {
            db_T[d * N + n] = db[n * D + d];
        }
    }
    CUDA_CHECK(cudaMalloc(&d_db_T_, bytes));
    CUDA_CHECK(cudaMemcpy(d_db_T_, db_T.data(), bytes, cudaMemcpyHostToDevice));
}

Engine::~Engine() {
    cudaFree(d_db_);
    cudaFree(d_db_T_);
}

void Engine::search(const float* query, int k, int kernel,
                    int* out_indices, float* out_scores) {
    // Copy query to device
    float* d_query;
    CUDA_CHECK(cudaMalloc(&d_query, d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query, query, d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Allocate scores on device
    float* d_scores;
    CUDA_CHECK(cudaMalloc(&d_scores, n_ * sizeof(float)));

    // Launch kernel -- switch use_tiled to compare both versions
    // use_tiled: 0 = global, 1 = shared, 2 = coalesced
    if (kernel == 2) {
        int threads = TILE_SIZE_COALESCED;
        int blocks = (n_ + threads - 1) / threads;
        cosine_shared_coalesced<<<blocks, threads>>>(d_db_T_, d_query, d_scores, n_, d_);
    } else if (kernel == 1) {
        // Tiled kernel: loads query tiles into shared memory
        // Reduces global memory traffic at the cost of sync overhead
        int threads = TILE_SIZE;
        int blocks  = (n_ + threads - 1) / threads;
        cosine_shared<<<blocks, threads>>>(d_db_, d_query, d_scores, n_, d_);
    } else {
        // Naive kernel: each thread reads query from global memory independently
        int threads = 256;
        int blocks  = (n_ + threads - 1) / threads;
        cosine_global<<<blocks, threads>>>(d_db_, d_query, d_scores, n_, d_);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Top-K selection using Thrust
    thrust::device_vector<int> d_indices(n_);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // Sort scores descending, carrying indices
    thrust::sort_by_key(
        thrust::device_pointer_cast(d_scores),
        thrust::device_pointer_cast(d_scores + n_),
        d_indices.begin(),
        thrust::greater<float>()
    );

    // Copy top-K back to host
    CUDA_CHECK(cudaMemcpy(out_scores, d_scores, k * sizeof(float),
                          cudaMemcpyDeviceToHost));
    thrust::copy(d_indices.begin(), d_indices.begin() + k, out_indices);

    cudaFree(d_query);
    cudaFree(d_scores);
}
