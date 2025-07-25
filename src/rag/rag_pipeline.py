import yaml
import json
from pathlib import Path
from groq import Groq
import faiss
import numpy as np
from huggingface_hub import InferenceClient

class RAGPipeline:
    def __init__(
        self,
        config_path: Path = Path("config.yaml"),
        index_path: Path = Path("embeddings/faiss_bangla.index"),
        meta_path: Path = Path("embeddings/faiss_bangla.index.meta.json"),
        hf_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        groq_model: str = "llama3-70b-8192"
    ):
        # Load configs
        cfg = yaml.safe_load(config_path.read_text())
        self.hf_token = cfg["hf_api"]["token"]
        self.groq_key = cfg["rag_api"]["key"]
        self.groq_model = groq_model
        
        # Inference client for embeddings
        self.embedder = InferenceClient(model=hf_model, token=self.hf_token)
        
        # FAISS index + metadata
        self.index = faiss.read_index(str(index_path))
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        
        # Normalize index vectors if you haven’t
        # (FAISS stores normalized vectors for IndexFlatIP)
        
        # Groq client
        self.groq = Groq(api_key=self.groq_key)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.embedder.feature_extraction([query])[0]
        # Average token embeddings if resp[0] is token-level
        if isinstance(resp[0], list):
            resp = [sum(col)/len(col) for col in zip(*resp)]
        vec = np.array(resp, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, top_k: int = 5):
        q_vec = self.embed_query(query)
        D, I = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            chunk = self.meta[idx]
            results.append({
                "id": chunk["chunk_id"],
                "score": float(score),
                "text": chunk["text"]
            })
        return results

    def generate_answer(self, query: str, contexts: list[dict]):
        # Build prompt
        prompt = "Use the following contexts (Bangla) to answer the user’s question.\n\n"
        for i, c in enumerate(contexts, 1):
            prompt += f"[{i}] {c['text']}\n\n"
        prompt += f"Question: {query}\nAnswer in Bangla:"
        
        # Call Groq chat completion
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512
        )
        return response.choices[0].message.content

    def __call__(self, query: str, top_k: int = 5):
        contexts = self.retrieve(query, top_k)
        answer = self.generate_answer(query, contexts)
        return {"answer": answer, "contexts": contexts}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    pipeline = RAGPipeline()
    out = pipeline(args.query, args.top_k)
    print("Answer:\n", out["answer"])
    print("\nContexts:")
    for c in out["contexts"]:
        print(f"- [{c['id']}] score={c['score']:.3f}")
        print(f"  {c['text'][:200]}…\n")
