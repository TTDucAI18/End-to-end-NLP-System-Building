import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.evaluate import evaluate


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_new")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--index", default="models/rag_index.joblib")
    parser.add_argument("--output", default="system_outputs/system_output_1.txt")
    parser.add_argument("--debug", default="reports/debug_predictions.jsonl")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    questions = data_dir / "test" / "questions.txt"
    references = data_dir / "test" / "reference_answers.txt"

    run(
        [
            sys.executable,
            "-m",
            "src.rag_system",
            "build",
            "--data-dir",
            args.data_dir,
            "--raw-dir",
            args.raw_dir,
            "--index",
            args.index,
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.rag_system",
            "predict",
            "--index",
            args.index,
            "--questions",
            str(questions),
            "--output",
            args.output,
            "--debug",
            args.debug,
            "--top-k",
            str(args.top_k),
        ]
    )

    predictions = Path(args.output).read_text(encoding="utf-8").splitlines()
    reference_lines = references.read_text(encoding="utf-8").splitlines()
    metrics = evaluate(predictions, reference_lines)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
