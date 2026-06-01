import faiss
import numpy as np
import time
import sys
sys.path.insert(0, '.')
import cuda_search

embeddings = np.load('data/embeddings.npy')
engine = cuda_search.Engine(embeddings)
query = embeddings[0]

# FAISS GPU
index_cpu = faiss.IndexFlatIP(embeddings.shape[1])
index_cpu.add(embeddings)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Benchmark
runs = 100
times_cuda, times_faiss = [], []
for _ in range(runs):
    t0 = time.perf_counter()
    engine.search(query, 10, 2)
    times_cuda.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    index.search(query.reshape(1,-1), 10)
    times_faiss.append((time.perf_counter() - t0) * 1000)

print(f"CUDA coalesced: {np.median(times_cuda):.3f}ms")
print(f"FAISS GPU:      {np.median(times_faiss):.3f}ms")