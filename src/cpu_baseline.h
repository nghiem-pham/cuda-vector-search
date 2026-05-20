#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// CPU sequential vector search -- used as:
// 1. Correctness baseline to verify CUDA output
// 2. Performance baseline for benchmarking

struct Result {
    int   index;
    float score;
};

// Cosine similarity between two L2-normalized vectors
// If both vectors are pre-normalized, this reduces to dot product
inline float cosine_similarity(const float* a, const float* b, int D) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < D; d++) {
        dot    += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}

// Brute-force top-K search on CPU
// db:    float array [N x D]
// query: float array [D]
// Returns top-K results sorted by score descending
inline std::vector<Result> cpu_search(const float* db, const float* query,
                                       int N, int D, int k) {
    std::vector<float> scores(N);
    for (int i = 0; i < N; i++) {
        scores[i] = cosine_similarity(db + i * D, query, D);
    }

    // Partial sort to get top-K indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<Result> results(k);
    for (int i = 0; i < k; i++) {
        results[i] = { indices[i], scores[indices[i]] };
    }
    return results;
}
