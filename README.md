# cuda-vector-search

A GPU-accelerated vector similarity search engine built from scratch in CUDA C++, with a semantic search demo using real text embeddings.

## What it does

Computes cosine similarity between a query vector and a large embedding database in parallel across thousands of GPU threads. Two kernels are implemented and benchmarked:

- `cosine_global` -- each thread reads query from global memory
- `cosine_shared` -- query tiles cached in shared memory to reduce global memory traffic
- `cosine_shared_coalesced` -- shared memory + column-major layout for coalesced db access

## Benchmark Results

Tested on RTX 3060, D=384, K=10.

| Method                | N=10k         | N=100k        | N=1M          |
|-----------------------|---------------|---------------|---------------|
| CPU sequential        | 4.44ms        | 39.62ms       | 410.71ms      |
| CUDA global           | 0.43ms (10x)  | 2.51ms (16x)  | 25.60ms (16x) |
| CUDA shared           | 0.41ms (11x)  | 1.36ms (29x)  | 12.30ms (33x) |
| CUDA shared+coalesced | 0.42ms (10x)  | 1.16ms (34x)  | 9.65ms (43x) |

Recall@10: 100% across all dataset sizes.

Speedup increases with N -- coalesced memory access becomes more effective at larger scale as memory bandwidth becomes the dominant bottleneck.

## Structure

```
cuda-vector-search/
├── src/
│   ├── kernels/
│   │   ├── cosine_global.cu    # baseline kernel
│   │   └── cosine_shared.cu    # shared memory optimized kernel
│   ├── engine.h                # Engine interface
│   ├── engine.cu               # Engine implementation + top-K selection
│   ├── cpu_baseline.h          # CPU sequential baseline
│   └── main.cpp                # benchmark entry point
├── python/
│   ├── bindings.cpp            # pybind11 Python bindings
│   ├── demo.py                 # semantic search CLI demo
│   └── embed_msmarco.py        # MS MARCO embedding pipeline
│   └── bench.py        # Our CUDA vs FAISS benchmark
├── Makefile
└── requirements.txt
```

## Setup

Requires NVIDIA GPU with CUDA 12.x.

```bash
# Build CUDA engine
make engine

# Install Python dependencies
pip install -r requirements.txt

# Build pybind11 bindings
make bindings

# Embed MS MARCO dataset (takes ~10 min)
python3 python/embed_msmarco.py --limit 50000 --output data/
```

## Usage

**Benchmark CPU vs CUDA:**
```bash
./engine 1000000 384 10
```

**Semantic search demo:**
```bash
python3 python/demo.py --query "how does computer memory work" --k 5
python3 python/demo.py --interactive
```

**Benchmark CUDA vs FAISS:**
```bash
python3 python/bench.py
```

## Stack

- CUDA C++, Thrust (top-K selection)
- pybind11, Python, sentence-transformers
- MS MARCO dataset
- Makefile

## Team

Nghiem Pham -- CUDA kernels (global, shared, coalesced), benchmarking, Makefile

Tiffany Karki -- pybind11 Python bindings, MS MARCO embedding pipeline, semantic search demo, FAISS comparison
