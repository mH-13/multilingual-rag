# src/api/admin.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import subprocess

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF into the knowledge base.
    Steps:
      1. Save to data/raw/
      2. OCR via ocr_parser → data/processed/raw_text_<name>.txt
      3. (Optional) PDF textual extraction for fallback
      4. Clean → data/processed/clean_text_<name>.txt
      5. Chunk → data/processed/chunks_<name>/
      6. Embed → embeddings/chunks_<name>.json
      7. Index → embeddings/faiss_<name>.index
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed.")
    name = Path(file.filename).stem

    # 1) Save raw PDF
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / file.filename
    with raw_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    # define output paths
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True, parents=True)
    raw_txt = processed / f"raw_text_{name}.txt"
    clean_txt = processed / f"clean_text_{name}.txt"
    chunks_dir = processed / f"chunks_{name}"
    embeds_json = Path("embeddings") / f"chunks_{name}.json"
    index_path = Path("embeddings") / f"faiss_{name}.index"
    embeds_json.parent.mkdir(exist_ok=True, parents=True)

    # 2–7) pipeline steps
    cmds = [
        # OCR step
        ["python", "src/extract/ocr_parser.py", str(raw_path), str(raw_txt)],
        # (Optional) fallback text extraction
        # ["python", "src/extract/pdf_parser.py", str(raw_path), "--method", "pymupdf", "--out", str(raw_txt)],
        # Clean
        ["python", "src/preprocess/cleaner.py", str(raw_txt), str(clean_txt)],
        # Chunk
        ["python", "src/chunking/char_chunker.py", str(clean_txt), str(chunks_dir), "--max-chars", "2000", "--overlap", "200"],
        # Embed
        ["python", "src/embeddings/embedder.py", "--chunks-dir", str(chunks_dir), "--out", str(embeds_json), "--batch-size", "10"],
        # Index
        ["python", "src/vector_store/indexer.py", "--embeddings", str(embeds_json), "--index-out", str(index_path)]
    ]

    try:
        for cmd in cmds:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Ingestion failed at step: {' '.join(e.cmd)}")

    return {"detail": f"Successfully ingested '{file.filename}' as '{name}'"}
