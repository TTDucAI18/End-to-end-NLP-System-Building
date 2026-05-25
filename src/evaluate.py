import argparse
import json
import re
import string
from pathlib import Path


_PUNCT = set(string.punctuation + "“”‘’…–—·")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common: dict[str, int] = {}
    for token in ref_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    return overlap / len(ref_tokens)


def split_references(line: str) -> list[str]:
    refs = [part.strip() for part in line.split(";") if part.strip()]
    return refs or [line.strip()]


def score_one(prediction: str, reference_line: str) -> dict[str, float]:
    references = split_references(reference_line)
    return {
        "exact_match": max(exact_match(prediction, ref) for ref in references),
        "f1": max(f1_score(prediction, ref) for ref in references),
        "answer_recall": max(answer_recall(prediction, ref) for ref in references),
    }


def evaluate(predictions: list[str], reference_lines: list[str]) -> dict[str, float]:
    if len(predictions) != len(reference_lines):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(predictions)} != {len(reference_lines)}"
        )
    rows = [score_one(pred, ref) for pred, ref in zip(predictions, reference_lines)]
    count = len(rows)
    if count == 0:
        return {"exact_match": 0.0, "f1": 0.0, "answer_recall": 0.0, "count": 0.0}
    return {
        "exact_match": sum(row["exact_match"] for row in rows) / count,
        "f1": sum(row["f1"] for row in rows) / count,
        "answer_recall": sum(row["answer_recall"] for row in rows) / count,
        "count": float(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    predictions = Path(args.predictions).read_text(encoding="utf-8").splitlines()
    references = Path(args.references).read_text(encoding="utf-8").splitlines()
    metrics = evaluate(predictions, references)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
