# End-to-End NLP System Building Assignment 2

This repository contains the code for a retrieval-augmented QA system for VNU/UET questions.

The repository intentionally excludes generated models, artifacts, and submission data files. Put the assignment data in the expected local layout before running.

Expected local data layout:

```text
data_new/
├── train/
│   ├── questions.txt
│   └── reference_answers.txt
└── test/
    ├── questions.txt
    └── reference_answers.txt

data/
└── raw/
    └── *.txt
```

## Run

```powershell
python run_experiment.py
```

Outputs:

- `system_outputs/system_output_1.txt`
- `reports/metrics.json`
- `reports/debug_predictions.jsonl`

## Components

- Hybrid retriever: word TF-IDF plus character TF-IDF.
- Training memory: annotated train QA pairs are indexed as supervised task data.
- Public knowledge resource: cleaned chunks from `data/raw` can be indexed as supporting documents.
- Reader: extractive answer selection with short-answer post-processing.

Additional robustness track:

```powershell
python -m src.preprocess_raw --raw-dir data\raw --output processed\raw_docs.jsonl
python -m src.raw_rag train --processed processed\raw_docs.jsonl --data-dir data_distinct --model models\raw_rag_distinct.joblib
python -m src.qwen_reader --rag-model models\raw_rag_distinct.joblib --questions data_distinct\test\questions.txt --output reports\qwen_raw_output.txt
```

The Qwen reader is optional and CPU-heavy. It uses retrieved raw context and does not read test reference answers during prediction.

## Submission Format

The Canvas zip should use the PDF-required structure:

```text
ANDREWID/
├── report.pdf
├── github_url.txt
├── contributions.md
├── data/
│   ├── test/
│   │   ├── questions.txt
│   │   ├── reference_answers.txt
│   ├── train/
│   │   ├── questions.txt
│   │   ├── reference_answers.txt
├── system_outputs/
│   ├── system_output_1.txt
│   ├── system_output_2.txt
│   ├── system_output_3.txt
└── README.md
```
