import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path

def extract_with_pdfplumber(pdf_path: Path) -> str:
    """Extract all text from PDF using pdfplumber."""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n\n".join(text)

def extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract all text from PDF using PyMuPDF."""
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)

def save_text(text: str, out_path: Path):
    """Save extracted text to a UTF‑8 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved raw text to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract PDF text")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--method",
        choices=["plumber", "pymupdf"],
        default="plumber",
        help="Extraction library to use",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../../data/processed/raw_text.txt"),
        help="Output .txt file",
    )
    args = parser.parse_args()

    if args.method == "plumber":
        raw = extract_with_pdfplumber(args.pdf)
    else:
        raw = extract_with_pymupdf(args.pdf)

    save_text(raw, args.out)
