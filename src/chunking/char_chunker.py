# src/chunking/char_chunker.py

from pathlib import Path

def chunk_by_chars(
    text: str, 
    max_chars: int = 2000, 
    overlap: int = 200
):
    chunks = []
    start = 0
    text_len = len(text)
    chunk_id = 0

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk_text = text[start:end]
        chunks.append({"chunk_id": chunk_id, "text": chunk_text})
        chunk_id += 1
        # advance start by max_chars - overlap
        start += (max_chars - overlap)
    return chunks

def save_chunks(chunks, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch in chunks:
        fn = f"chunk_{ch['chunk_id']:04d}.txt"
        (out_dir / fn).write_text(ch["text"], encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {out_dir}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="Clean text file")
    p.add_argument("outdir", type=Path, help="Dir for chunk files")
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    text = args.input.read_text(encoding="utf-8")
    chunks = chunk_by_chars(text, args.max_chars, args.overlap)
    save_chunks(chunks, args.outdir)
