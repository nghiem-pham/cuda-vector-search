"""
Semantic search demo using the CUDA engine.

Usage:
    python demo.py --query "how to handle errors in async code" --k 5
    python demo.py --interactive
"""

import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import cuda_search  # built pybind11 module

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR   = Path("data")

def load_data():
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    chunks     = (DATA_DIR / "chunks.txt").read_text(encoding="utf-8").split("\n\n")
    return embeddings, chunks

def search(engine, model, chunks, query_text: str, k: int = 5):
    # Embed query
    q_vec = model.encode([query_text], normalize_embeddings=True)[0].astype("float32")

    # Search
    indices, scores = engine.search(q_vec, k, 2)

    return [
        {"rank": i + 1, "score": float(scores[i]), "text": chunks[indices[i]]}
        for i in range(k)
    ]

def print_results(results):
    for r in results:
        print(f"\n[{r['rank']}] score={r['score']:.4f}")
        print(f"    {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       default=None, help="Single query")
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    embeddings, chunks = load_data()
    print(f"Database: {len(chunks)} chunks, {embeddings.shape[1]}-dim vectors")

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading CUDA engine...")
    engine = cuda_search.Engine(embeddings)
    print(f"Engine ready: N={engine.N}, D={engine.D}\n")

    if args.interactive:
        while True:
            try:
                query = input("Query (Ctrl+C to exit): ").strip()
                if not query:
                    continue
                results = search(engine, model, chunks, query, args.k)
                print_results(results)
            except KeyboardInterrupt:
                break
    elif args.query:
        results = search(engine, model, chunks, args.query, args.k)
        print_results(results)
    else:
        # Default demo queries
        queries = [
            "how to handle errors in async code",
            "parallel programming with threads",
            "GPU memory optimization techniques",
        ]
        for q in queries:
            print(f"\nQuery: '{q}'")
            print_results(search(engine, model, chunks, q, args.k))

if __name__ == "__main__":
    main()
