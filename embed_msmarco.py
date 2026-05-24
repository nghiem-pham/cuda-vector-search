"""
embed_msmarco.py -- embed MS MARCO passages dataset
Usage: python3 python/embed_msmarco.py --limit 50000 --output data/

MS MARCO is a large-scale dataset of web search queries and passages.
"""

import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=50000, help="Max number of passages")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MS MARCO (up to {args.limit} passages)...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")

    chunks = []
    for item in dataset:
        for p in item["passages"]["passage_text"]:
            chunks.append(p.strip())
        if len(chunks) >= args.limit:
            break
    chunks = chunks[:args.limit]
    print(f"Passages: {len(chunks)}")

    print(f"Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "chunks.txt").write_text("\n\n".join(chunks))
    print(f"Saved {len(chunks)} passages, shape {embeddings.shape}")

if __name__ == "__main__":
    main()