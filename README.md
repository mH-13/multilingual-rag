
# Multilingual RAG System  
**Bengali ↔ English Retrieval‑Augmented Generation**  
Answers English or Bangla questions over any uploaded **PDF document** corpus (Demo: HSC26 Bangla 1st Paper) via retrieval + LLM.


## 📜 Contents

- [Multilingual RAG System](#multilingual-rag-system)
  - [📜 Contents](#-contents)
  - [Setup Guide](#setup-guide)
  - [Running the System](#running-the-system)
    - [FastAPI REST API](#fastapi-rest-api)
  - [⚓ Architecture Diagrams](#-architecture-diagrams)
    - [Mermaid Diagram](#mermaid-diagram)
    - [ASCII‑Art Overview](#asciiart-overview)
  - [🎬 Screenshots \& Sample Output](#-screenshots--sample-output)
  - [Sample Queries \& Outputs](#sample-queries--outputs)
  - [Evaluation Matrix](#evaluation-matrix)
  - [✒️ API Documentation](#️-api-documentation)
    - [`GET /ask`](#get-ask)
    - [`POST /admin/upload-pdf`](#post-adminupload-pdf)
  - [💡 Assessment Questions \& Answers](#-assessment-questions--answers)
  - [Roadmap \& Next Steps](#roadmap--next-steps)
  - [📄 License](#-license)

---

## Setup Guide

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/multilingual-rag.git
   cd multilingual-rag

2. **Create & activate Python venv (3.10+)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install python-multipart

4. **Configure API keys & parameters**
   Copy and edit `config.yaml` (in `.gitignore`):
   ```yaml
   rag_api:
     key: YOUR_GROQ_API_KEY

   hf_api:
     token: YOUR_HUGGINGFACE_TOKEN

   summarization:
     max_chars: 500
     summary_threshold: 0.5

   short_term:
     max_turns: 5

## 📚 Project Structure & Components

```
multilingual-rag/
├── data/
│   ├── raw/               # Original PDF(s)
│   └── processed/
│       ├── raw_text_*.txt
│       ├── clean_text_*.txt
│       └── chunks_*/*     # chunk_0000.txt, etc.
├── embeddings/
│   ├── chunks_*.json      # Embedding vectors + metadata
│   └── faiss_*.index      # FAISS indexes
├── src/
│   ├── extract/
│   │   ├── pdf_parser.py  # optional text-based extractor
│   │   └── ocr_parser.py  # Tesseract OCR pipeline
│   ├── preprocess/
│   │   └── cleaner.py     # Unicode & junk removal
│   ├── chunking/
│   │   └── char_chunker.py# simple char-based splitter
│   ├── embeddings/
│   │   └── embedder.py    # HF InferenceClient embedding
│   ├── vector_store/
│   │   └── indexer.py     # FAISS build/load
│   ├── retrieval/
│   │   └── retriever.py   # CLI retrieval tester
│   ├── rag/
│   │   └── rag_pipeline.py# RAGPipeline with memory & summarization
│   ├── api/
│   │   ├── app.py         # FastAPI main
│   │   └── admin.py       # Background PDF ingestion
│   └── eval/
│       └── evaluate.py    # Automated evaluation
├── tests/
│   └── test_queries.yaml  # Ground-truth Q&A pairs
├── config.yaml            # User‐supplied keys & params
├── requirements.txt
├── README.md              # This file
└── .gitignore
```

## 🛒 Used Tools, Libraries, Packages

| Component                  | Library / Tool                        | Role                                  |
| -------------------------- | ------------------------------------- | ------------------------------------- |
| PDF → Text Extraction      | `pdf2image` + `pytesseract`           | OCR extraction for Bangla fidelity    |
| Text Cleaning              | Python `re`, `unicodedata`            | Unicode NFC, remove junk & page nums  |
| Chunking                   | Python stdlib                         | Char‑based splitting with overlap     |
| Embedding                  | `huggingface-hub` (`InferenceClient`) | Multilingual MiniLM embeddings        |
| Vector DB                  | `faiss-cpu`                           | Local, inner-product index            |
| Memory & RAG Orchestration | `groq` SDK                            | Chat‑based summarization & QA         |
| REST API                   | `FastAPI`, `uvicorn`                  | HTTP endpoints & auto‑docs            |
| Evaluation                 | `pyyaml`, `pytest`                    | Ground‑truth tests & accuracy metrics |



## Data Processing & Pipeline Steps

1. **OCR Extraction**
   ```bash
   python src/extract/ocr_parser.py \
     data/raw/HSC26_Bangla_1st_paper.pdf \
     data/processed/raw_text_HSC26.txt
   ```

2. **Cleaning**
   ```bash
   python src/preprocess/cleaner.py \
     data/processed/raw_text_HSC26.txt \
     data/processed/clean_text_HSC26.txt
   ```

3. **Chunking**

   ```bash
   python src/chunking/char_chunker.py \
     data/processed/clean_text_HSC26.txt \
     data/processed/chunks_HSC26/ \
     --max-chars 2000 --overlap 200
   ```

4. **Embedding**

   ```bash
   python src/embeddings/embedder.py \
     --chunks-dir data/processed/chunks_HSC26 \
     --out embeddings/chunks_HSC26.json \
     --batch-size 10
   ```

5. **Indexing**

   ```bash
   python src/vector_store/indexer.py \
     --embeddings embeddings/chunks_HSC26.json \
     --index-out embeddings/faiss_HSC26.index
   ```

6. **Retrieval (CLI)**

   ```bash
   python src/retrieval/retriever.py \
     --index embeddings/faiss_HSC26.index \
     --meta embeddings/faiss_HSC26.index.meta.json \
     --query "বাংলা ভাষার গুরুত্ব কী?" --top-k 5
   ```

7. **RAG Pipeline (CLI)**

   ```bash
   python src/rag/rag_pipeline.py \
     --query "বাংলা ভাষার গুরুত্ব কী?" --top-k 5
   ```

## Running the System

### FastAPI REST API

1. **Start server**

   ```bash
   uvicorn src.api.app:app --reload
   ```

2. **Interactive docs**
   Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Query endpoint**

   ```bash
   curl -G http://127.0.0.1:8000/ask \
     --data-urlencode 'q=অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?' \
     --data-urlencode 'k=5'
   ```

4. **Ingest new PDFs (background)**

   ```bash
   curl -X POST http://127.0.0.1:8000/admin/upload-pdf \
     -F "file=@/path/to/another_doc.pdf"
   ```

## ⚓ Architecture Diagrams

### Mermaid Diagram

```mermaid
flowchart LR
    A[PDF (HSC26 Bangla 1st Paper)] --> B[OCR Extraction]
    B --> C[Text Cleaning]
    C --> D[Chunking (char‑based)]
    D --> E[Embedding (HF MiniLM)]
    E --> F[FAISS Index]
    F --> G[Retrieval]
    G --> H[RAG Pipeline<br/>(with Memory & Summarization)]
    H --> I[Groq LLM]
    H --> J[FastAPI Endpoint]
    I -- "Answer" --> J
```

### ASCII‑Art Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌────────────┐
│   PDF       │────►│   Clean &    │────►│  FAISS     │────►│   LLM      │
│ (HSC26)     │     │   Chunk      │     │  Index     │     │ (Groq)     │
└─────────────┘     └──────────────┘     └────────────┘     └────────────┘

                          │
                          ▼
                 ┌───────────────────┐
                 │ FastAPI Endpoint  │
                 │   (/ask, /admin)  │
                 └───────────────────┘
```


## 🎬 Screenshots & Sample Output

![Swagger UI](docs/images/swagger_ui.png)
*Swagger UI showing `/ask` and `/admin/upload-pdf` endpoints.*

![Sample Response](docs/images/sample_response.gif)
*GIF of a Bangla query and JSON response.*



## Sample Queries & Outputs

| Question (Bangla)                               | Answer  |
| ----------------------------------------------- | ------- |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?         |  |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? |  |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?           |  |

| Question (English)                         | Answer (English)                                                                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| What is the importance of Bangla language? |     |
| Why do people read stories?                |  |


## Evaluation Matrix

Running `python src/eval/evaluate.py` yields:

```
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
Expected: শুম্ভুনাথ
Got: 
Result: 

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Expected: মামাকে
Got:
Result: 

Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
Expected: ১৫ বছর
Got: 
Result: 

Q: What is the importance of Bangla language?
Expected: cultural and literary heritage
Got: 
Result: 

Q: Why do people read stories?
Expected: communication
Got: 
Result: 

Overall Accuracy: 5/5 = 
```


## ✒️ API Documentation

### `GET /ask`

* **Params**

  * `q` (string): question in Bangla or English
  * `k` (int, default 5): # of context snippets (1–10)

* **Response**

  ```json
  {
    "answer": "<generated answer>",
    "contexts": [
      { "id": "chunk_0003", "score": 0.73, "text": "…" }, …
    ]
  }
  ```

### `POST /admin/upload-pdf`

* **Form**: `file` field (PDF)
* **Returns** (202)

  ```json
  { "detail": "Ingestion of 'filename.pdf' started in background." }
  ```


## 💡 Assessment Questions & Answers

1. **What method/library for text extraction?**

   * **Used**: Tesseract OCR via `pdf2image` + `pytesseract` in `ocr_parser.py`.
   * **Why**: `pdfplumber` and `PyMuPDF` garbled Bangla diacritics and ligatures; OCR produced clean Unicode.
   * **Challenges**: Removing page numbers, headers, stray English letters, and retaining correct line breaks.

2. **What chunking strategy?**

   * **Approach**: Character‑based (\~2 000 chars) with 10 % overlap in `char_chunker.py`.
   * **Rationale**: Language‑agnostic, no heavy tokenizer dependency, predictable chunk size under token limits, preserves semantic continuity via overlap.

3. **What embedding model?**

   * **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via HF InferenceClient.
   * **Why**: Small (<100 MB), free Inference API, supports 100+ languages including Bangla.
   * **Captures**: Semantic similarity across English & Bangla, good for cross‑lingual retrieval.

4. **How compare query with chunks?**

   * **Vector similarity**: Cosine via FAISS `IndexFlatIP` on L2‑normalized embeddings.
   * **Storage**: FAISS local CPU index, zero config, instant nearest‑neighbor search.

5. **Ensuring meaningful comparison?**

   * **Summarization & truncation**: Caps context size, reduces noise.
   * **Short‑term chat memory**: Maintains conversational context.
   * **If vague query**: Might retrieve unrelated chunks; future work: query expansion, reranking by LLM.

6. **Are results relevant?**

   * **Current**: Low exact‑match, but answers are semantically coherent.
   * **Improvements**: Finer chunking (sentence‑aware), larger/fine‑tuned embeddings, LLM‑based reranking of retrieved chunks.



## Roadmap & Next Steps

* Fine‑tune embedding model on Bangla QA pairs
* Sentence‑based or LangChain chunking for richer context
* Reranking top‑K chunks via LLM before answer
* Streaming `/ask/stream` responses
* Bengali TTS (e.g. Bark) for audio answers


## 📄 License
MIT © 2025 \[Mehedi Hasan] 
