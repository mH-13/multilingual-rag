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
      1. Save file to data/raw/
      2. OCR extract → data/processed/raw_text_<name>.txt
      3. Clean → data/processed/clean_text_<name>.txt
      4. Chunk → data/processed/chunks_<name>/
      5. Embed → embeddings/chunks_<name>.json
      6. Build index → embeddings/faiss_<name>.index + .meta.json
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed.")
    name = Path(file.filename).stem

    # 1. Save raw PDF
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / file.filename
    with raw_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    # Define paths for each step
    processed_dir = Path("data/processed")
    raw_txt = processed_dir / f"raw_text_{name}.txt"
    clean_txt = processed_dir / f"clean_text_{name}.txt"
    chunks_dir = processed_dir / f"chunks_{name}"
    embeds_json = Path("embeddings") / f"chunks_{name}.json"
    index_path = Path("embeddings") / f"faiss_{name}.index"

    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeds_json.parent.mkdir(parents=True, exist_ok=True)

    # 2–6. Run pipeline commands sequentially
    cmds = [
        ["python", "src/extract/pdf_parser.py", str(raw_path), "--method", "ocr", "--out", str(raw_txt)],
        ["python", "src/preprocess/cleaner.py", str(raw_txt), str(clean_txt)],
        ["python", "src/chunking/char_chunker.py", str(clean_txt), str(chunks_dir), "--max-chars", "2000", "--overlap", "200"],
        ["python", "src/embeddings/embedder.py", "--chunks-dir", str(chunks_dir), "--out", str(embeds_json), "--batch-size", "10"],
        ["python", "src/vector_store/indexer.py", "--embeddings", str(embeds_json), "--index-out", str(index_path)]
    ]

    try:
        for cmd in cmds:
            subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Ingestion failed at step: {' '.join(e.cmd)}")

    return {"detail": f"Successfully ingested PDF '{file.filename}' into the KB."}
