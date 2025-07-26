# src/api/admin.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import shutil
import subprocess

router = APIRouter(prefix="/admin", tags=["admin"])

def ingest_pipeline(raw_path: Path, name: str):
    """
    Run the full extract→clean→chunk→embed→index pipeline for a PDF.
    This function is executed in the background.
    """
    processed = Path("data/processed")
    # Define intermediate and output paths
    raw_txt     = processed / f"raw_text_{name}.txt"
    clean_txt   = processed / f"clean_text_{name}.txt"
    chunks_dir  = processed / f"chunks_{name}"
    embeds_json = Path("embeddings") / f"chunks_{name}.json"
    index_path  = Path("embeddings") / f"faiss_{name}.index"

    # Ensure directories exist
    processed.mkdir(exist_ok=True, parents=True)
    embeds_json.parent.mkdir(exist_ok=True, parents=True)

    # Define each CLI step
    cmds = [
        # 1) OCR extraction
        ["python", "src/extract/ocr_parser.py", str(raw_path), str(raw_txt)],
        # 2) Clean text
        ["python", "src/preprocess/cleaner.py", str(raw_txt), str(clean_txt)],
        # 3) Chunk text
        ["python", "src/chunking/char_chunker.py",
            str(clean_txt), str(chunks_dir),
            "--max-chars", "2000", "--overlap", "200"],
        # 4) Embed chunks
        ["python", "src/embeddings/embedder.py",
            "--chunks-dir", str(chunks_dir),
            "--out", str(embeds_json),
            "--batch-size", "10"],
        # 5) Build FAISS index
        ["python", "src/vector_store/indexer.py",
            "--embeddings", str(embeds_json),
            "--index-out", str(index_path)]
    ]

    # Execute each step, raising if any fail
    for cmd in cmds:
        subprocess.run(cmd, check=True)

@router.post("/upload-pdf", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF and start ingestion in the background.
    Returns immediately with HTTP 202.
    """
    # Validate file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Save the raw PDF
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / file.filename
    with raw_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    # Schedule the background ingestion task
    name = raw_path.stem
    background_tasks.add_task(ingest_pipeline, raw_path, name)

    return {"detail": f"Ingestion of '{file.filename}' started in background."}
