#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "engine.h"
#include "cpu_baseline.h"

// L2-normalize a vector in place
void normalize(float* v, int D) {
    float norm = 0.0f;
    for (int d = 0; d < D; d++) norm += v[d] * v[d];
    norm = std::sqrt(norm);
    if (norm > 1e-8f)
        for (int d = 0; d < D; d++) v[d] /= norm;
}

int main(int argc, char* argv[]) {
    // Default values
    int N = 100000;  // database size
    int D = 384;     // embedding dimension
    int K = 10;      // top-K

    // Parse arguments: ./engine N D K
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) D = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    std::cout << "N=" << N << " D=" << D << " K=" << K << "\n\n";

    // Generate random normalized vectors
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> db(N * D), query(D);
    for (auto& x : db)    x = dist(rng);
    for (auto& x : query) x = dist(rng);

    // Normalize all vectors
    for (int i = 0; i < N; i++) normalize(db.data() + i * D, D);
    normalize(query.data(), D);

    // ── CPU baseline ──────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cpu_results = cpu_search(db.data(), query.data(), N, D, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU sequential: " << cpu_ms << "ms\n";


    // ── CUDA engine ───────────────────────────────
    Engine engine(db.data(), N, D);
    std::vector<int>   cuda_indices(K);
    std::vector<float> cuda_scores(K);

    // Warmup
    engine.search(query.data(), K, false, cuda_indices.data(), cuda_scores.data());

    auto t2 = std::chrono::high_resolution_clock::now();
    engine.search(query.data(), K, false, cuda_indices.data(), cuda_scores.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double global_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "CUDA global:    " << global_ms << "ms"
          << "  (" << cpu_ms / global_ms << "x)\n";

    // ── CUDA shared kernel ────────────────────────
    // Warmup
    engine.search(query.data(), K, true, cuda_indices.data(), cuda_scores.data());

    auto t4 = std::chrono::high_resolution_clock::now();
    engine.search(query.data(), K, true, cuda_indices.data(), cuda_scores.data());
    auto t5 = std::chrono::high_resolution_clock::now();
    double shared_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    std::cout << "CUDA shared:    " << shared_ms << "ms"
          << "  (" << cpu_ms / shared_ms << "x)\n";

    // ── Correctness check ─────────────────────────
    int matches = 0;
    for (auto& r : cpu_results) {
        for (int i = 0; i < K; i++) {
            if (cuda_indices[i] == r.index) {
                matches++; break;
            }
        }
    }
    std::cout << "Recall@" << K << ": " << matches << "/" << K
              << " (" << (100.0 * matches / K) << "%)\n";
    std::cout << "Global speedup: " << cpu_ms / global_ms << "x\n";
    std::cout << "Shared speedup: " << cpu_ms / shared_ms << "x\n";

    return 0;
}
