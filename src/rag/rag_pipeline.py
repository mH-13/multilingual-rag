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
        # --- Load configuration ---
        cfg = yaml.safe_load(config_path.read_text())

        # Hugging Face & Groq credentials
        self.hf_token = cfg["hf_api"]["token"]
        self.groq_key = cfg["rag_api"]["key"]
        self.groq_model = groq_model

        # Summarization/truncation settings
        summ_cfg = cfg["summarization"]
        self.max_chars = summ_cfg["max_chars"]
        self.summary_threshold = summ_cfg["summary_threshold"]

        # Short‑term memory settings
        st_cfg = cfg.get("short_term", {})
        self.max_turns = st_cfg.get("max_turns", 5)
        # Each turn has two messages (user + assistant)
        self.max_history_messages = self.max_turns * 2
        # Initialize empty history
        self.history: list[dict] = []

        # Initialize embedding client (HF)
        self.embedder = InferenceClient(model=hf_model, token=self.hf_token)

        # Load FAISS index and metadata
        self.index = faiss.read_index(str(index_path))
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        # Initialize Groq client
        self.groq = Groq(api_key=self.groq_key)

    def embed_query(self, query: str) -> np.ndarray:
        # Embed the user query via HF feature-extraction
        resp = self.embedder.feature_extraction([query])[0]
        # If token-level embeddings, average across tokens
        if isinstance(resp[0], list):
            resp = [sum(col) / len(col) for col in zip(*resp)]
        vec = np.array(resp, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, top_k: int = 5):
        # Retrieve top_k chunk contexts
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
        # Perform LLM-based summarization on a chunk
        prompt = f"Summarize this text in {target_lang} in 1–2 sentences:\n\n{text}"
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system",  "content": "You are a helpful summarizer."},
                {"role": "user",    "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def generate_answer(self, query: str, contexts: list[dict]):
        # Detect language of query
        is_bangla = any("\u0980" <= ch <= "\u09FF" for ch in query)
        lang = "Bangla" if is_bangla else "English"

        # Prepare each snippet: truncate and optionally summarize
        snippets = []
        for c in contexts:
            # 1) Truncate to max_chars
            txt = c["text"][:self.max_chars].rsplit("\n", 1)[0] + "…"
            # 2) Summarize if needed
            if not is_bangla:
                # English query: always summarize into English
                txt = self._summarize_chunk(txt, "English")
            else:
                # Bangla query: summarize only if low similarity
                if c["score"] < self.summary_threshold:
                    txt = self._summarize_chunk(txt, "Bangla")
            snippets.append(txt)

        # Build full prompt including short‑term history
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        # Append memory: past user+assistant turns
        messages.extend(self.history)
        # Add the retrieval context as a new user message
        prompt_text = f"Use these contexts to answer the question in {lang}:\n\n"
        for i, s in enumerate(snippets, 1):
            prompt_text += f"[{i}] {s}\n\n"
        prompt_text += f"Question: {query}\nAnswer in {lang}:"
        messages.append({"role": "user", "content": prompt_text})

        # Call Groq chat completion with history
        response = self.groq.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            max_tokens=512
        )
        return response.choices[0].message.content

    def __call__(self, query: str, top_k: int = 5):
        # Manage short‑term memory size before retrieval
        # We only keep the last max_history_messages entries
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]

        # 1) Append current user query to history
        self.history.append({"role": "user", "content": query})

        # 2) Retrieve contexts and generate answer
        contexts = self.retrieve(query, top_k)
        answer = self.generate_answer(query, contexts)

        # 3) Append assistant response to history
        self.history.append({"role": "assistant", "content": answer})

        # Return result
        return {"answer": answer, "contexts": contexts}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Run RAG pipeline")
    p.add_argument("--query", required=True, type=str, help="User question")
    p.add_argument("--top-k", type=int, default=5, help="Number of contexts")
    args = p.parse_args()

    pipeline = RAGPipeline()
    result = pipeline(args.query, args.top_k)

    print("Answer:\n", result["answer"])
    print("\nContexts:")
    for c in result["contexts"]:
        print(f"- [{c['id']}] score={c['score']:.3f}")
        print(c["text"][:200] + "…\n")
