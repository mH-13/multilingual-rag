# src/embeddings/embedder.py

import yaml
import json
from pathlib import Path
from huggingface_hub import InferenceClient

def load_configs(path: Path = Path("config.yaml")):
    cfg = yaml.safe_load(path.read_text())
    groq_key = cfg["embeddings_api"]["key"]
    hf_token = cfg["hf_api"]["token"]
    return groq_key, hf_token

def embed_texts_hf(texts: list[str], hf_token: str, repo_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    client = InferenceClient(model=repo_id, token=hf_token)
    resp = client.feature_extraction(texts)
    return resp 

def embed_chunks(chunk_dir: Path, out_path: Path, batch_size: int = 10):
    """Embed chunks via HF Inference API and save to embeds JSON."""
    _, hf_token = load_configs()
    embeds = []
    files = sorted(chunk_dir.glob("chunk_*.txt"))
    # batch uploads of N chunks at a time
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        texts = [f.read_text(encoding="utf-8") for f in batch]
        vecs = embed_texts_hf(texts, hf_token)
        for fn, vec in zip(batch, vecs):
            embeds.append({
                "chunk_id": fn.stem,
                "text": fn.read_text(encoding="utf-8"),
                "vector": vec.tolist() if hasattr(vec, "tolist") else vec
            })
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(embeds), encoding="utf-8")
    print(f"Embedded {len(embeds)} chunks → {out_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Embed text chunks via Hugging Face Inference API")
    p.add_argument("--chunks-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=10)
    args = p.parse_args()

    embed_chunks(args.chunks_dir, args.out, args.batch_size)
