# multilingual-rag
### Bilingual Retrieval-Augmented Generation over the HSC26 Bangla 1st Paper  
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)  
[![FastAPI](https://img.shields.io/badge/api-fastapi-teal.svg)](https://fastapi.tiangolo.com)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview  
multilingual-rag is a **lightweight, production-ready** system that answers **English or Bengali questions** by retrieving relevant excerpts from the **HSC26 Bangla 1st Paper** PDF and generating concise, grounded responses.  
The entire stack runs **locally on CPU** or can be switched to any cloud LLM in one line.

---

## Quick Start (Local CPU)

| Step | Command |
|---|---|
| 1. Clone | `git clone https://github.com/<your-username>/benglish-rag.git && cd benglish-rag` |
| 2. Install | `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| 3. Add PDF | Place `HSC26_Bangla_1st_paper.pdf` in `data/raw/` |
| 4. Build DB | `python -m src.pdf_extractor && python -m src.chunker && python -m src.vector_store` |
| 5. Launch API | `uvicorn api.main:app --reload` |
| 6. Test | `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"কল্যাণীর প্রকৃত বয়স কত ছিল?"}'` |

---

## 🧰 Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌────────────┐
│   PDF       │────►│   Clean &    │────►│  FAISS     │────►│   LLM      │
│  (HSC26)    │     │   Chunk      │     │  Index     │     │ (OpenAI/   │
└─────────────┘     └──────────────┘     └────────────┘     │  Gemini …) │
                                                           └────────────┘
```

---

## Evaluation (Dev Set)

| Metric | Value |
|---|---|
| Exact Match (Bengali) | 87.5 % (7/8) |
| Exact Match (English) | 75 % (3/4) |
| Avg Latency (CPU) | 2.1 s |
| Tokens / Query | ~180 |

Run `python tests/evaluate.py` to reproduce.

---

## API Reference

### `POST /ask`
| Parameter | Type | Description |
|---|---|---|
| `question` | string | English or Bengali question |
| `k` (opt) | int | # of retrieved chunks (default 3) |

**Example Request**  
```json
{
  "question": "অনুপেমর ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "k": 2
}
```

**Example Response**  
```json
{
  "answer": "শুম্ভুনাথ",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Interactive docs: http://localhost:8000/docs

---

## Sample Queries

| Question | Answer |
|---|---|
| `কল্যাণীর প্রকৃত বয়স কত ছিল?` | `১৫ বছর` |
| `Who is referred to as the ‘hero’ in Anupam’s speech?` | `Shumbhnath` |
| `অনুপেম কোন গ্রামে যায়?` | `শান্তিপুর` |

---

## ⚙️ Tech Stack

| Layer | Tool | Rationale |
|---|---|---|
| PDF → Text | `pymupdf` | Fast, accurate layout |
| Chunking | `langchain RecursiveCharacterTextSplitter` | 500-char, 75-overlap, sentence-aware |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | 100 MB, bilingual |
| Vector DB | `FAISS` (CPU) | Zero-config, local |
| LLM (default) | `ChatOpenAI` (`gpt-4o-mini`) | Cheap, fast, bilingual |
| API | `FastAPI` | Auto-generated docs |
| Memory | `deque` window | Last 3 QA pairs |
| Evaluation | `ragas` + exact-match | Relevance & groundedness |

---

## Switching LLM Provider

Edit `.env` and one line in `src/generator.py`:

| Provider | `.env` Key | Code Change |
|---|---|---|
| **Gemini** | `GOOGLE_API_KEY` | `from langchain_google_genai import ChatGoogleGenerativeAI` |
| **Claude** | `ANTHROPIC_API_KEY` | `from langchain_anthropic import ChatAnthropic` |
| **Groq** | `GROQ_API_KEY` | `from langchain_openai import ChatOpenAI` + custom `base_url` |

No vector DB rebuild required.

---

## Development

1. **Lint & Format**
   ```bash
   pip install ruff
   ruff format src/ api/ tests/
   ```
2. **Tests**
   ```bash
   pytest tests/
   ```
3. **Docker (optional)**
   ```bash
   docker build -t benglish-rag .
   docker run -p 8000:8000 --env-file .env benglish-rag
   ```

---

## Roadmap
- [ ] Fine-tune embedding on Bangla sentence pairs  
- [ ] Metadata filtering (chapter, page)  
- [ ] Streaming responses (`/ask/stream`)  
- [ ] Bengali TTS (`bark`) for audio answers  

---

## Contributing
PRs welcome! Please include tests and run `ruff format`.

---

## 📄 License  
MIT © 2024 [Mehedi Hasan]  
Feel free to use in academic and commercial projects.
```
