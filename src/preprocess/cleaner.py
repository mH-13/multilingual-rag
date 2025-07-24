import re
import unicodedata
from pathlib import Path

BANGla_CHAR_RANGE = (
    "\u0980-\u09FF"  # Bangla Unicode block
)

def normalize_unicode(text: str) -> str:
    # NFC normalization composes combined letters correctly
    return unicodedata.normalize("NFC", text)

def remove_non_bangla(text: str) -> str:
    # Keep Bangla letters, Bengali punctuation, and common whitespace/punct
    pattern = rf"[^{BANGla_CHAR_RANGE}\s\u0964\u0965\u200C\u200D\.,!?;:“”‘’\"\'\-—\(\)]"
    return re.sub(pattern, "", text)

def fix_spacing(text: str) -> str:
    # Ensure space after punctuation if missing
    text = re.sub(r"([।!?;,”»])([^\s])", r"\1 \2", text)
    # Remove multiple spaces/tabs/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse multiple blank lines
    return text.strip()

def remove_page_numbers(text: str) -> str:
    # Remove lines that are purely digits or page headers like “Page 1”
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\d+\s*", line):
            continue
        if re.match(r"\s*Page\s*\d+", line, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines)

def clean_text(raw_path: Path, clean_path: Path):
    raw = raw_path.read_text(encoding="utf-8")
    text = normalize_unicode(raw)
    text = remove_page_numbers(text)
    text = remove_non_bangla(text)
    text = fix_spacing(text)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(text, encoding="utf-8")
    print(f"Cleaned text saved to {clean_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Clean extracted Bangla text")
    p.add_argument("input", type=Path, help="Raw text file")
    p.add_argument("output", type=Path, help="Clean text output")
    args = p.parse_args()
    clean_text(args.input, args.output)
