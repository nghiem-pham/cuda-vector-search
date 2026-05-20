# cuda-vector-search
 
A GPU-accelerated vector similarity search engine built from scratch in CUDA C++.
 
## What it does
 
Computes cosine similarity between a query vector and a large embedding database in parallel across thousands of GPU threads. Two kernels are implemented and benchmarked:
 
- `cosine_global` -- each thread reads query from global memory
- `cosine_shared` -- query tiles cached in shared memory to reduce global memory traffic
## Benchmark Results
 
Tested on RTX 3060, D=384, K=10.
 
| Method         | N=10k        | N=100k        | N=1M          |
|----------------|--------------|---------------|---------------|
| CPU sequential | 2.46ms       | 23.65ms       | 235.89ms      |
| CUDA global    | 0.39ms (6x)  | 2.27ms (10x)  | 24.71ms (10x) |
| CUDA shared    | 0.36ms (7x)  | 1.30ms (18x)  | 11.74ms (20x) |
 
Recall@10: 100% across all dataset sizes.
 
Speedup increases with N -- shared memory optimization becomes more effective at larger scales as memory bandwidth becomes the dominant bottleneck.
 
## Structure
 
```
src/
├── kernels/
│   ├── cosine_global.cu    # baseline kernel
│   └── cosine_shared.cu    # shared memory optimized kernel
├── engine.hpp              # Engine interface
├── engine.cu               # Engine implementation + top-K selection
├── cpu_baseline.hpp        # CPU sequential baseline
└── main.cpp                # benchmark entry point
```
 
## Build
 
Requires NVIDIA GPU with CUDA 12.x.
 
```bash
make engine
./engine 1000000 384 10
```
 
## Stack
 
- CUDA C++
- Thrust (top-K selection)
- CMake / Makefile