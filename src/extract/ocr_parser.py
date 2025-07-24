from pdf2image import convert_from_path
import pytesseract
from pathlib import Path

def ocr_pdf(pdf_path: Path, out_txt: Path):
    imgs = convert_from_path(str(pdf_path), dpi=300)
    text_chunks = []
    for i, img in enumerate(imgs):
        txt = pytesseract.image_to_string(img, lang="ben")
        text_chunks.append(txt)
    full = "\n\n".join(text_chunks)
    out_txt.write_text(full, encoding="utf-8")
    print(f"OCR’d text to {out_txt}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    ocr_pdf(args.pdf, args.out)
