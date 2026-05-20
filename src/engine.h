#pragma once
#include <cstdint>

// Vector search engine interface

class Engine {
public:
    // Load N vectors of dimension D onto GPU
    // db: float array of shape [N x D], row-major
    Engine(const float* db, int N, int D);
    ~Engine();

    // Find top-K nearest neighbors by cosine similarity
    // query:       float array of shape [D], already L2-normalized
    // k:           number of results to return
    // out_indices: int array of shape [k], caller-allocated
    // out_scores:  float array of shape [k], caller-allocated
    void search(const float* query, int k, bool use_tiled,
                int* out_indices, float* out_scores);

    int N() const { return n_; }
    int D() const { return d_; }

private:
    float* d_db_;   // device pointer to database
    int    n_;
    int    d_;
};