# src/rag/rag_pipeline.py

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
        cfg = yaml.safe_load(config_path.read_text())
        self.hf_token = cfg["hf_api"]["token"]
        self.groq_key = cfg["rag_api"]["key"]
        self.groq_model = groq_model

        summ_cfg = cfg["summarization"]
        self.max_chars = summ_cfg["max_chars"]
        self.summary_threshold = summ_cfg["summary_threshold"]

        self.embedder = InferenceClient(model=hf_model, token=self.hf_token)
        self.index = faiss.read_index(str(index_path))
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.groq = Groq(api_key=self.groq_key)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.embedder.feature_extraction([query])[0]
        if isinstance(resp[0], list):
            resp = [sum(col) / len(col) for col in zip(*resp)]
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

    def _summarize_chunk(self, text: str, target_lang: str) -> str:
        prompt = f"Summarize the following text in {target_lang} in 1–2 sentences:\n\n{text}"
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful summarizer."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def generate_answer(self, query: str, contexts: list[dict]):
        # Detect language
        is_bangla = any("\u0980" <= ch <= "\u09FF" for ch in query)
        lang = "Bangla" if is_bangla else "English"

        # If English query: translate/summarize *all* contexts into English
        snippets = []
        for c in contexts:
            txt = c["text"][: self.max_chars].rsplit("\n", 1)[0] + "…"

            if not is_bangla:
                # for English, always summarize to English
                txt = self._summarize_chunk(txt, "English")
            else:
                # for Bangla, summarize only if similarity low
                if c["score"] < self.summary_threshold:
                    txt = self._summarize_chunk(txt, "Bangla")

            snippets.append(txt)

        # Build prompt
        prompt = f"Use these contexts to answer the question in {lang}:\n\n"
        for i, s in enumerate(snippets, 1):
            prompt += f"[{i}] {s}\n\n"
        prompt += f"Question: {query}\nAnswer in {lang}:"

        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=512
        )
        return response.choices[0].message.content

    def __call__(self, query: str, top_k: int = 5):
        # If English, suggest a smaller top_k for speed, but still enforce <=10
        if not any("\u0980" <= ch <= "\u09FF" for ch in query):
            top_k = min(top_k, 3)
        contexts = self.retrieve(query, top_k)
        answer = self.generate_answer(query, contexts)
        return {"answer": answer, "contexts": contexts}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Run RAG pipeline")
    p.add_argument("--query", required=True, type=str)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    pipeline = RAGPipeline()
    out = pipeline(args.query, args.top_k)
    print("Answer:\n", out["answer"])
    print("\nContexts:")
    for c in out["contexts"]:
        print(f"- [{c['id']}] score={c['score']:.3f}")
        print(c["text"][:200], "…\n")
