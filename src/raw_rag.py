import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


from src.evaluate import f1_score, split_references


CODE_RE = re.compile(r"\b(?:CN|QH|TH|VNU|UET|HUS|ULIS|UMP|UED|VJU|IS|FEPN|FCE|ICEMA)[A-Z0-9._/-]*\b", re.IGNORECASE)
MONEY_RE = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*đồng)?(?:/năm)?\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
URL_RE = re.compile(r"https?://[^\s)]+|[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s)]*)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


@dataclass
class RawDoc:
    doc_id: str
    text: str
    source: str
    kind: str


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))


def overlap_ratio(a: str, b: str) -> float:
    left = {tok for tok in tokenize(a) if len(tok) > 1}
    right = tokenize(b)
    if not left:
        return 0.0
    return len(left & right) / len(left)


class RawRag:
    def __init__(self, word_weight: float = 0.55, char_weight: float = 0.45) -> None:
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.92,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.feature_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=40000,
        )
        self.selector: LogisticRegression | None = None
        self.fact_memory: dict[str, list[tuple[str, str]]] = {}
        self.docs: list[RawDoc] = []
        self.word_matrix = None
        self.char_matrix = None

    def fit_index(self, docs: list[RawDoc]) -> None:
        self.docs = docs
        texts = [doc.text for doc in docs]
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

    def retrieve(self, question: str, top_k: int = 25) -> list[tuple[int, float]]:
        if self.word_matrix is None or self.char_matrix is None:
            raise ValueError("Index has not been fitted")
        qw = self.word_vectorizer.transform([question])
        qc = self.char_vectorizer.transform([question])
        scores = (
            self.word_weight * (self.word_matrix @ qw.T).toarray().ravel()
            + self.char_weight * (self.char_matrix @ qc.T).toarray().ravel()
        )
        for idx, doc in enumerate(self.docs):
            if doc.kind.startswith("fact"):
                scores[idx] += 0.035
        indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        indices = indices[np.argsort(-scores[indices])]
        return [(int(idx), float(scores[idx])) for idx in indices if scores[idx] > 0]

    def train_selector(self, data_dir: Path, top_k: int = 20) -> dict:
        questions = load_lines(data_dir / "train" / "questions.txt")
        references = load_lines(data_dir / "train" / "reference_answers.txt")
        self.fact_memory = build_fact_memory(questions, references)
        candidate_texts: list[str] = []
        labels: list[int] = []
        positives = 0
        negatives = 0

        for question, reference_line in zip(questions, references):
            refs = split_references(reference_line)
            for doc_idx, score in self.retrieve(question, top_k=top_k):
                doc = self.docs[doc_idx]
                candidates = candidate_answers(question, doc.text)
                if not candidates:
                    candidates = [best_sentence(question, doc.text)]
                for candidate in candidates[:8]:
                    label = int(max(f1_score(candidate, ref) for ref in refs) >= 0.55)
                    candidate_texts.append(pair_text(question, doc.text, candidate))
                    labels.append(label)
                    positives += label
                    negatives += 1 - label

        if positives == 0:
            raise ValueError("No positive selector examples found")
        matrix = self.feature_vectorizer.fit_transform(candidate_texts)
        self.selector = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
        )
        self.selector.fit(matrix, labels)
        return {
            "examples": len(labels),
            "positives": positives,
            "negatives": negatives,
            "fact_memory": sum(len(v) for v in self.fact_memory.values()),
        }

    def answer(self, question: str, top_k: int = 25) -> dict:
        memory_answer = answer_from_fact_memory(question, self.fact_memory)
        if memory_answer:
            return {
                "answer": memory_answer,
                "retrieved": [{"doc_id": "trained_fact_memory", "source": "data_distinct/train", "kind": "trained_fact"}],
            }

        retrieved = self.retrieve(question, top_k=top_k)
        pending = []
        for rank, (doc_idx, retrieval_score) in enumerate(retrieved):
            doc = self.docs[doc_idx]
            candidates = candidate_answers(question, doc.text)
            if not candidates:
                candidates = [best_sentence(question, doc.text)]
            for candidate in candidates[:10]:
                features = pair_text(question, doc.text, candidate)
                pending.append((features, candidate, doc, retrieval_score, rank))

        if not pending:
            return {"answer": "", "retrieved": []}

        selector_scores = np.zeros(len(pending), dtype=float)
        if self.selector is not None:
            matrix = self.feature_vectorizer.transform([item[0] for item in pending])
            selector_scores = self.selector.predict_proba(matrix)[:, 1]

        rows = []
        for (features, candidate, doc, retrieval_score, rank), selector_score in zip(pending, selector_scores):
            final_score = score_candidate(
                question, candidate, doc.kind, retrieval_score, float(selector_score), rank
            )
            rows.append((final_score, candidate, doc, retrieval_score, float(selector_score)))

        rows.sort(key=lambda item: (-item[0], len(item[1].split()), len(item[1])))
        best = rows[0]
        return {
            "answer": best[1],
            "retrieved": [
                {
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "kind": doc.kind,
                    "retrieval_score": retrieval_score,
                    "selector_score": selector_score,
                }
                for _, _, doc, retrieval_score, selector_score in rows[:5]
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RawRag":
        return joblib.load(path)


def pair_text(question: str, context: str, candidate: str) -> str:
    return f"question: {question}\ncontext: {context[:900]}\nanswer: {candidate}"


def candidate_answers(question: str, text: str) -> list[str]:
    q = question.lower()
    candidates: list[str] = []

    if any(key in q for key in ["website", "url", "trang web", "địa chỉ đăng ký"]):
        candidates.extend(match.group(0).rstrip(".,;") for match in URL_RE.finditer(text))
    if "học phí" in q:
        candidates.extend(match.group(0) for match in MONEY_RE.finditer(text))
    if any(key in q for key in ["mã", "viết tắt", "tên viết tắt"]):
        candidates.extend(match.group(0).upper() for match in CODE_RE.finditer(text))
    if any(key in q for key in ["năm nào", "thành lập", "vào năm"]):
        candidates.extend(match.group(0) for match in YEAR_RE.finditer(text))
    if any(key in q for key in ["bao nhiêu", "mấy", "chỉ tiêu", "điểm"]):
        candidates.extend(match.group(0) for match in NUMBER_RE.finditer(text))

    for sentence in split_sentences(text):
        if overlap_ratio(question, sentence) >= 0.20:
            candidates.append(trim_answer(question, sentence))

    return unique(candidates)


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", ". ")
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text)
    return [normalize_space(part).strip(" .") for part in parts if len(part.split()) >= 3]


def trim_answer(question: str, sentence: str) -> str:
    sentence = normalize_space(sentence).strip(" .,:;")
    if len(sentence.split()) <= 20:
        return sentence
    lowered = sentence.lower()
    q = question.lower()
    markers = []
    if "là gì" in q or "là ai" in q or "tên" in q:
        markers.extend([" là ", " có tên tiếng anh là ", " viết tắt là "])
    if "thuộc" in q:
        markers.append(" thuộc ")
    if "gồm" in q or "những" in q:
        markers.extend([" gồm ", " bao gồm "])
    for marker in markers:
        idx = lowered.find(marker)
        if idx >= 0:
            tail = sentence[idx + len(marker) :].strip(" .,:;")
            if 1 <= len(tail.split()) <= 18:
                return tail
    return " ".join(sentence.split()[:20]).strip(" .,:;")


def best_sentence(question: str, text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_space(text)[:180]
    return trim_answer(question, max(sentences, key=lambda sentence: overlap_ratio(question, sentence)))


def unique(candidates: list[str]) -> list[str]:
    seen = set()
    output = []
    for candidate in candidates:
        candidate = normalize_space(candidate).strip(" .,:;")
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def score_candidate(
    question: str,
    candidate: str,
    kind: str,
    retrieval_score: float,
    selector_score: float,
    rank: int,
) -> float:
    q = question.lower()
    score = retrieval_score + 0.9 * selector_score - 0.02 * rank
    words = candidate.split()
    if len(words) <= 8:
        score += 0.05
    if len(words) > 25:
        score -= 0.15
    if kind.startswith("fact"):
        score += 0.06
    if "học phí" in q and MONEY_RE.search(candidate):
        score += 0.18
    if ("mã" in q or "viết tắt" in q) and CODE_RE.fullmatch(candidate):
        score += 0.18
    if ("năm nào" in q or "thành lập" in q) and YEAR_RE.fullmatch(candidate):
        score += 0.14
    if ("website" in q or "url" in q) and URL_RE.fullmatch(candidate):
        score += 0.14
    return score


INTENT_PATTERNS = {
    "fee": ["học phí", "mức học phí"],
    "code": ["mã xét tuyển", "mã ngành", "mã của ngành", "mã là gì"],
    "quota": ["chỉ tiêu", "tuyển bao nhiêu", "bao nhiêu sinh viên"],
    "english_name": ["tên tiếng anh", "tiếng anh là gì", "tên đầy đủ tiếng anh"],
    "abbrev": ["tên viết tắt", "viết tắt là gì"],
    "year": ["năm nào", "vào năm nào", "năm bao nhiêu"],
    "field": ["lĩnh vực gì", "định hướng lĩnh vực", "đào tạo lĩnh vực"],
    "role": ["nhiệm vụ gì", "vai trò gì", "chức năng", "trách nhiệm gì"],
    "time": ["thời gian nào", "diễn ra vào", "khi nào"],
    "place": ["ở đâu", "địa điểm nào", "tại đâu"],
    "what": ["là gì", "gì?"],
}


STOP_ENTITY_WORDS = {
    "hãy",
    "cho",
    "biết",
    "xin",
    "vui",
    "lòng",
    "ngành",
    "của",
    "tại",
    "trường",
    "đại",
    "học",
    "uet",
    "đhqghn",
    "vnu",
    "là",
    "gì",
    "bao",
    "nhiêu",
    "có",
    "mức",
    "tên",
    "tiếng",
    "anh",
    "viết",
    "tắt",
    "mã",
    "xét",
    "tuyển",
    "chỉ",
    "tiêu",
}


def detect_intent(question: str) -> str | None:
    q = question.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(pattern in q for pattern in patterns):
            return intent
    return None


def entity_key(question: str) -> str:
    text = question.lower()
    text = re.sub(r"\b(hãy cho biết|xin cho biết|cho biết|vui lòng cho biết)\b", " ", text)
    for patterns in INTENT_PATTERNS.values():
        for pattern in patterns:
            text = text.replace(pattern, " ")
    tokens = re.findall(r"[\wÀ-ỹ]+", text)
    kept = [tok for tok in tokens if tok not in STOP_ENTITY_WORDS and len(tok) > 1]
    return " ".join(kept)


def answer_value(reference_line: str) -> str:
    refs = split_references(reference_line)
    return min(refs, key=lambda item: (len(item.split()), len(item)))


def build_fact_memory(questions: list[str], references: list[str]) -> dict[str, list[tuple[str, str]]]:
    memory: dict[str, list[tuple[str, str]]] = {}
    for question, reference in zip(questions, references):
        intent = detect_intent(question)
        key = memory_key(question)
        if not intent or len(key) < 2:
            continue
        memory.setdefault(intent, []).append((key, answer_value(reference)))
    return memory


def answer_from_fact_memory(question: str, memory: dict[str, list[tuple[str, str]]]) -> str | None:
    intent = detect_intent(question)
    key = memory_key(question)
    if not intent or not key or intent not in memory:
        return None
    best_score = 0.0
    best_answer = None
    for train_key, answer in memory[intent]:
        score = char_ngram_f1(key, train_key)
        if score > best_score:
            best_score = score
            best_answer = answer
    threshold = 0.72
    if intent in {"fee", "code", "quota", "english_name", "abbrev"}:
        threshold = 0.66
    if best_score >= threshold:
        return best_answer
    return None


def memory_key(question: str) -> str:
    text = question.lower()
    text = re.sub(r"\b(hãy cho biết|xin cho biết|cho biết|vui lòng cho biết)\b", " ", text)
    text = re.sub(r"[^\wÀ-ỹ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def char_ngrams(text: str, n: int = 4) -> set[str]:
    text = f" {text.lower()} "
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def char_ngram_f1(left: str, right: str) -> float:
    a = char_ngrams(left)
    b = char_ngrams(right)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(b)
    return 2 * precision * recall / (precision + recall)


def build_docs(processed_path: Path) -> list[RawDoc]:
    rows = load_jsonl(processed_path)
    return [
        RawDoc(
            doc_id=str(row["id"]),
            text=row["text"],
            source=row.get("source", ""),
            kind=row.get("kind", "chunk"),
        )
        for row in rows
    ]


def train_command(args: argparse.Namespace) -> None:
    docs = build_docs(Path(args.processed))
    model = RawRag(word_weight=args.word_weight, char_weight=args.char_weight)
    model.fit_index(docs)
    selector_stats = model.train_selector(Path(args.data_dir), top_k=args.train_top_k)
    model.save(Path(args.model))
    stats = {
        "processed": args.processed,
        "data_dir": args.data_dir,
        "docs": len(docs),
        "word_weight": args.word_weight,
        "char_weight": args.char_weight,
        **selector_stats,
    }
    Path(args.model).with_suffix(".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    model = RawRag.load(Path(args.model))
    questions = load_lines(Path(args.questions))
    outputs = []
    debug_rows = []
    for question in questions:
        result = model.answer(question, top_k=args.top_k)
        outputs.append(result["answer"])
        debug_rows.append({"question": question, **result})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(outputs) + "\n", encoding="utf-8")
    if args.debug:
        Path(args.debug).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.debug).open("w", encoding="utf-8") as handle:
            for row in debug_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--processed", default="processed/raw_docs.jsonl")
    train.add_argument("--data-dir", default="data_distinct")
    train.add_argument("--model", default="models/raw_rag_distinct.joblib")
    train.add_argument("--word-weight", type=float, default=0.55)
    train.add_argument("--char-weight", type=float, default=0.45)
    train.add_argument("--train-top-k", type=int, default=20)
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--model", default="models/raw_rag_distinct.joblib")
    predict.add_argument("--questions", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--debug")
    predict.add_argument("--top-k", type=int, default=30)
    predict.set_defaults(func=predict_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
