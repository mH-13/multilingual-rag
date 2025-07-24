import faiss
import numpy as np
import json
import yaml
from pathlib import Path
from huggingface_hub import InferenceClient

def load_config(path: Path = Path("config.yaml")):
    cfg = yaml.safe_load(path.read_text())
    return cfg["hf_api"]["token"]

def embed_query(query: str, hf_token: str, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Embed the user query via the same HF pipeline used for chunks.
    """
    client = InferenceClient(model=model, token=hf_token)
    resp = client.feature_extraction([query])  # returns [[...]]
    vec = resp[0]
    # Average token embeddings if needed (for sentence-transformers it's already a single vector)
    if isinstance(vec[0], list):
        # token-level lists → average
        vec = [sum(col)/len(col) for col in zip(*vec)]
    return np.array(vec, dtype="float32")

def load_index(index_path: Path, meta_path: Path):
    # load FAISS index
    index = faiss.read_index(str(index_path))
    # load metadata
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return index, meta

def retrieve_top_k(
    query: str,
    index: faiss.Index,
    meta: list[dict],
    hf_token: str,
    top_k: int = 5
):
    # 1) embed
    q_vec = embed_query(query, hf_token)
    # 2) normalize (since we used inner-product on normalized vectors)
    faiss.normalize_L2(q_vec.reshape(1, -1))
    # 3) search
    D, I = index.search(q_vec.reshape(1, -1), top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        item = meta[idx]
        results.append({
            "chunk_id": item["chunk_id"],
            "score": float(dist),
            "text": item["text"][:300] + "…"  # preview first 300 chars
        })
    return results

def main():
    import argparse

    p = argparse.ArgumentParser("Retrieve top‑k chunks for a query")
    p.add_argument("--index", type=Path, required=True,
                   help="FAISS index file (.index)")
    p.add_argument("--meta", type=Path, required=True,
                   help="Metadata JSON (.meta.json)")
    p.add_argument("--query", type=str, required=True,
                   help="User query (Bangla or English)")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    hf_token = load_config()
    index, meta = load_index(args.index, args.meta)
    results = retrieve_top_k(args.query, index, meta, hf_token, args.top_k)

    print(f"Top {args.top_k} chunks for query: “{args.query}”\n")
    for r in results:
        print(f"[{r['chunk_id']}] (score={r['score']:.4f})")
        print(r["text"])
        print("-" * 80)

if __name__ == "__main__":
    main()
