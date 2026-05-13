# cuda-vector-search
 
A GPU-accelerated vector similarity search engine built from scratch in CUDA, wrapped into a semantic search demo using real text embeddings.
 
## What it does
 
Given a text query, the system finds the most semantically similar documents from a large embedding database. Similarity computation runs on the GPU using custom CUDA kernels, achieving significant speedup over CPU baseline.
 
## Architecture
 
```
Text query
    → sentence-transformers (embed)
    → 384-dim float vector
    → CUDA engine (cosine similarity × N vectors)
    → top-K results
```
