import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


NOISE_PATTERNS = [
    r"^read more about\b",
    r"^xem thêm\b",
    r"^trang chủ\b",
    r"^menu\b",
    r"^search\b",
    r"^facebook\b",
    r"^twitter\b",
    r"^copyright\b",
    r"^liên hệ\b",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return True
    if len(lowered) <= 2:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def clean_document(text: str) -> str:
    text = normalize_text(text)
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if is_noise_line(line):
            continue
        lines.append(line)
    return normalize_text("\n".join(lines))


def fingerprint(text: str) -> str:
    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def sentence_split(text: str) -> list[str]:
    text = text.replace("\n", ". ")
    parts = re.split(r"(?<=[.!?])\s+|(?<=。)\s+", text)
    return [part.strip(" .") for part in parts if len(part.split()) >= 5]


def chunk_sentences(sentences: list[str], max_words: int = 170, overlap_sentences: int = 1) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(item.split()) for item in current)
        current.append(sentence)
        current_words += words
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk.split()) >= 20]


def extract_fact_docs(text: str, source: str) -> list[dict]:
    facts: list[dict] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    # Admission table rows often contain code, major name, quota, and sometimes fee.
    for line in lines:
        if re.search(r"\bCN\d+\b", line):
            compact = re.sub(r"\s+", " ", line)
            if 8 <= len(compact.split()) <= 80:
                facts.append(
                    {
                        "id": "",
                        "source": source,
                        "kind": "fact_major",
                        "text": f"Thông tin tuyển sinh UET: {compact}",
                    }
                )

    # IELTS conversion tables are split across PDF text; keep local windows.
    for match in re.finditer(r"\bIELTS\b", joined, flags=re.IGNORECASE):
        start = max(0, match.start() - 450)
        end = min(len(joined), match.end() + 900)
        window = re.sub(r"\s+", " ", joined[start:end])
        if any(x in window.lower() for x in ["quy đổi", "thang điểm", "điểm cộng"]):
            facts.append(
                {
                    "id": "",
                    "source": source,
                    "kind": "fact_ielts",
                    "text": f"Bảng quy đổi/cộng điểm IELTS: {window}",
                }
            )

    # English names are high-value exact-answer facts.
    for sentence in sentence_split(text):
        lowered = sentence.lower()
        if "tên tiếng anh" in lowered or "tên viết tắt" in lowered or "viết tắt" in lowered:
            facts.append(
                {
                    "id": "",
                    "source": source,
                    "kind": "fact_name",
                    "text": sentence,
                }
            )

    return facts


def preprocess(raw_dir: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen = set()
    doc_count = 0
    chunk_count = 0
    fact_count = 0

    for path in sorted(raw_dir.glob("*.txt")):
        cleaned = clean_document(path.read_text(encoding="utf-8", errors="ignore"))
        if len(cleaned.split()) < 30:
            continue
        fp = fingerprint(cleaned)
        if fp in seen:
            continue
        seen.add(fp)
        doc_count += 1

        source = str(path.as_posix())
        facts = extract_fact_docs(cleaned, source)
        for fact in facts:
            fact["id"] = f"fact_{fact_count}"
            rows.append(fact)
            fact_count += 1

        chunks = chunk_sentences(sentence_split(cleaned))
        for chunk in chunks:
            rows.append(
                {
                    "id": f"chunk_{chunk_count}",
                    "source": source,
                    "kind": "chunk",
                    "text": chunk,
                }
            )
            chunk_count += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "raw_dir": str(raw_dir),
        "output": str(output),
        "documents_kept": doc_count,
        "chunks": chunk_count,
        "facts": fact_count,
        "total_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="processed/raw_docs.jsonl")
    args = parser.parse_args()
    stats = preprocess(Path(args.raw_dir), Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
