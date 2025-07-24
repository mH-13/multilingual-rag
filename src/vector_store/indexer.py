import faiss
import numpy as np
import json
from pathlib import Path

def build_faiss_index(embeddings_json: Path, index_path: Path):
    data = json.loads(embeddings_json.read_text(encoding="utf-8"))
    vectors = np.array([item["vector"] for item in data], dtype="float32")
    ids = [item["chunk_id"] for item in data]

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product for cosine sim if vectors are normalized
    faiss.normalize_L2(vectors)
    index.add(vectors)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save metadata (map index position → chunk_id & text)
    metadata = [{"chunk_id": ids[i], "text": data[i]["text"]} for i in range(len(ids))]
    Path(str(index_path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    print(f"Built FAISS index at {index_path} with {index.ntotal} vectors")



if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Build FAISS index from embeddings JSON")
    p.add_argument("--embeddings", type=Path, required=True,
                  help="JSON file with embeddings & metadata")
    p.add_argument("--index-out", type=Path, required=True,
                  help="Path to write FAISS index (.index) and metadata (.meta.json)")
    args = p.parse_args()

    build_faiss_index(args.embeddings, args.index_out)
