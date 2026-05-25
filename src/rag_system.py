import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


CODE_RE = re.compile(r"\b(?:CN|QH|TH|VNU|UET|HUS|ULIS|UMP|UED|IS|VJU)[A-Z0-9._-]*\b", re.IGNORECASE)
MONEY_RE = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*đồng)?(?:/năm)?\b|\b\d+\s*(?:triệu|tr)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)]+|[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s)]*)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


@dataclass
class Document:
    doc_id: str
    text: str
    answer: str
    source: str


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def split_answer(answer_line: str) -> list[str]:
    answers = [part.strip() for part in answer_line.split(";") if part.strip()]
    return answers or [answer_line.strip()]


def best_short_answer(answer_line: str) -> str:
    answers = split_answer(answer_line)
    return min(answers, key=lambda item: (len(item.split()), len(item)))


def clean_raw_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = normalize_space(line)
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"menu", "search", "login", "home"}:
            continue
        if len(line) < 25 and any(x in lowered for x in ["copyright", "facebook", "twitter"]):
            continue
        lines.append(line)
    return "\n".join(lines)


def chunk_text(text: str, size: int = 160, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def build_documents(data_dir: Path, raw_dir: Path | None = None, include_train_answers: bool = True) -> list[Document]:
    docs: list[Document] = []

    if include_train_answers:
        questions = load_lines(data_dir / "train" / "questions.txt")
        answers = load_lines(data_dir / "train" / "reference_answers.txt")
        if len(questions) != len(answers):
            raise ValueError("Train questions/reference_answers are not aligned")
        for idx, (question, answer_line) in enumerate(zip(questions, answers)):
            answer = best_short_answer(answer_line)
            text = f"Câu hỏi: {question}\nCâu trả lời: {answer_line}"
            docs.append(Document(f"train_qa_{idx}", text, answer, "train_qa"))

    if raw_dir and raw_dir.exists():
        for file_idx, path in enumerate(sorted(raw_dir.glob("*.txt"))):
            text = clean_raw_text(path.read_text(encoding="utf-8", errors="ignore"))
            for chunk_idx, chunk in enumerate(chunk_text(text)):
                docs.append(Document(f"raw_{file_idx}_{chunk_idx}", chunk, "", str(path)))

    if not docs:
        raise ValueError("No documents were built")
    return docs


class RagSystem:
    def __init__(
        self,
        word_weight: float = 0.58,
        char_weight: float = 0.42,
        qa_bonus: float = 0.08,
    ) -> None:
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.qa_bonus = qa_bonus
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.92,
            sublinear_tf=True,
            strip_accents=None,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.documents: list[Document] = []
        self.word_matrix = None
        self.char_matrix = None
        self.qa_questions: list[str] = []
        self.qa_answers: list[str] = []
        self.qa_word_matrix = None
        self.qa_char_matrix = None

    def fit(self, documents: list[Document]) -> None:
        self.documents = documents
        texts = [doc.text for doc in documents]
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)
        self.qa_questions = []
        self.qa_answers = []
        for doc in documents:
            if doc.source != "train_qa" or not doc.answer:
                continue
            question = doc.text.split("Câu trả lời:", 1)[0].replace("Câu hỏi:", "").strip()
            self.qa_questions.append(question)
            self.qa_answers.append(doc.answer)
        if self.qa_questions:
            self.qa_word_matrix = self.word_vectorizer.transform(self.qa_questions)
            self.qa_char_matrix = self.char_vectorizer.transform(self.qa_questions)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RagSystem":
        return joblib.load(path)

    def retrieve(self, question: str, top_k: int = 20) -> list[tuple[int, float]]:
        if self.word_matrix is None or self.char_matrix is None:
            raise ValueError("Index is not fitted")
        q_word = self.word_vectorizer.transform([question])
        q_char = self.char_vectorizer.transform([question])
        scores = (
            self.word_weight * (self.word_matrix @ q_word.T).toarray().ravel()
            + self.char_weight * (self.char_matrix @ q_char.T).toarray().ravel()
        )
        for i, doc in enumerate(self.documents):
            if doc.source == "train_qa":
                scores[i] += self.qa_bonus
        if top_k >= len(scores):
            indices = np.argsort(-scores)
        else:
            indices = np.argpartition(-scores, top_k)[:top_k]
            indices = indices[np.argsort(-scores[indices])]
        return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]

    def answer(self, question: str, top_k: int = 20) -> dict:
        direct_answer = self.direct_qa_answer(question)
        if direct_answer is not None:
            answer, score = direct_answer
            return {
                "answer": answer,
                "retrieved": [{"doc_id": "supervised_qa_memory", "source": "train_qa", "score": score}],
            }

        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {"answer": "", "retrieved": []}

        candidates: list[tuple[str, float, str]] = []
        for rank, (doc_idx, score) in enumerate(retrieved[:top_k]):
            doc = self.documents[doc_idx]
            rank_bonus = 1.0 / math.sqrt(rank + 1)
            if doc.answer:
                candidates.append((doc.answer, score + 0.18 * rank_bonus, doc.doc_id))
                for alt in split_answer(doc.text.split("Câu trả lời:", 1)[-1]):
                    candidates.append((alt, score + 0.14 * rank_bonus, doc.doc_id))
            for extracted in extract_from_text(question, doc.text):
                candidates.append((extracted, score + 0.10 * rank_bonus, doc.doc_id))

        if not candidates:
            best_doc = self.documents[retrieved[0][0]]
            candidates.append((fallback_sentence(question, best_doc.text), retrieved[0][1], best_doc.doc_id))

        answer = select_answer(question, candidates)
        return {
            "answer": answer,
            "retrieved": [
                {
                    "doc_id": self.documents[idx].doc_id,
                    "source": self.documents[idx].source,
                    "score": score,
                }
                for idx, score in retrieved[:5]
            ],
        }

    def direct_qa_answer(self, question: str) -> tuple[str, float] | None:
        literal_code = literal_code_answer(question)
        if literal_code:
            return literal_code, 1.0

        if self.qa_word_matrix is None or self.qa_char_matrix is None or not self.qa_answers:
            return None
        q_word = self.word_vectorizer.transform([question])
        q_char = self.char_vectorizer.transform([question])
        scores = (
            self.word_weight * (self.qa_word_matrix @ q_word.T).toarray().ravel()
            + self.char_weight * (self.qa_char_matrix @ q_char.T).toarray().ravel()
        )
        idx = int(np.argmax(scores))
        best_score = float(scores[idx])
        if best_score >= direct_threshold(question):
            return self.qa_answers[idx], best_score
        return None


def literal_code_answer(question: str) -> str | None:
    q = question.lower()
    match = re.search(r"\bCN\d+\b", question, re.IGNORECASE)
    if match and ("là mã xét tuyển" in q or "là mã ngành" in q):
        return match.group(0).upper()
    return None


def direct_threshold(question: str) -> float:
    q = question.lower()
    if any(key in q for key in ["ielts", "mã xét tuyển", "mã ngành", "học phí", "viết tắt"]):
        return 0.40
    if any(key in q for key in ["năm nào", "bao nhiêu", "thuộc", "tên tiếng anh"]):
        return 0.43
    return 0.47


def extract_from_text(question: str, text: str) -> list[str]:
    q = question.lower()
    candidates: list[str] = []

    if "url" in q or "website" in q or "trang web" in q or "địa chỉ web" in q:
        candidates.extend(match.group(0).rstrip(".,;") for match in URL_RE.finditer(text))
    if "học phí" in q or "bao nhiêu tiền" in q:
        candidates.extend(match.group(0) for match in MONEY_RE.finditer(text))
    if "mã" in q or "viết tắt" in q or "tên viết tắt" in q:
        candidates.extend(match.group(0).upper() for match in CODE_RE.finditer(text))
    if "năm nào" in q or "năm bao nhiêu" in q or "thành lập" in q:
        candidates.extend(match.group(0) for match in YEAR_RE.finditer(text))
    if "ngày" in q or "thời gian" in q or "diễn ra" in q:
        candidates.extend(match.group(0) for match in DATE_RE.finditer(text))
    if "bao nhiêu" in q or "mấy" in q:
        candidates.extend(match.group(0) for match in NUMBER_RE.finditer(text))

    for sentence in split_sentences(text):
        if lexical_overlap(question, sentence) >= 0.22:
            candidates.append(trim_sentence_answer(question, sentence))

    return unique_nonempty(candidates)


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", ". ")
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text)
    return [normalize_space(part) for part in parts if len(part.split()) >= 3]


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))


def lexical_overlap(question: str, sentence: str) -> float:
    q_tokens = {tok for tok in tokenize(question) if len(tok) > 1}
    s_tokens = tokenize(sentence)
    if not q_tokens:
        return 0.0
    return len(q_tokens & s_tokens) / len(q_tokens)


def trim_sentence_answer(question: str, sentence: str) -> str:
    sentence = re.sub(r"^Câu trả lời:\s*", "", sentence).strip()
    if len(sentence.split()) <= 18:
        return sentence
    q = question.lower()
    if "là gì" in q or "là ai" in q or "thuộc" in q:
        for marker in [" là ", " thuộc ", " gồm ", " bao gồm ", " tại "]:
            if marker in sentence.lower():
                idx = sentence.lower().find(marker)
                tail = sentence[idx + len(marker) :].strip(" .,:;")
                if 1 <= len(tail.split()) <= 18:
                    return tail
    return " ".join(sentence.split()[:18]).strip(" .,:;")


def fallback_sentence(question: str, text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_space(text)[:160]
    return max(sentences, key=lambda sent: lexical_overlap(question, sent))[:160].strip(" .,:;")


def unique_nonempty(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        item = normalize_space(item).strip(" .,:;")
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def select_answer(question: str, candidates: list[tuple[str, float, str]]) -> str:
    q = question.lower()
    scored = []
    for answer, score, doc_id in candidates:
        answer = normalize_space(answer).strip(" .,:;")
        if not answer:
            continue
        words = answer.split()
        adjusted = score
        if len(words) <= 8:
            adjusted += 0.08
        if len(words) > 24:
            adjusted -= 0.18
        if ("mã" in q or "viết tắt" in q) and CODE_RE.fullmatch(answer):
            adjusted += 0.22
        if ("học phí" in q or "bao nhiêu tiền" in q) and (MONEY_RE.search(answer) or "đồng" in answer.lower()):
            adjusted += 0.22
        if ("năm nào" in q or "thành lập" in q) and YEAR_RE.fullmatch(answer):
            adjusted += 0.18
        if ("website" in q or "url" in q) and URL_RE.fullmatch(answer):
            adjusted += 0.20
        if answer.lower().startswith("câu hỏi"):
            adjusted -= 0.50
        scored.append((adjusted, len(words), len(answer), answer, doc_id))
    if not scored:
        return ""
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return scored[0][3]


def build_command(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    docs = build_documents(Path(args.data_dir), raw_dir, include_train_answers=not args.no_train_qa)
    model = RagSystem(word_weight=args.word_weight, char_weight=args.char_weight, qa_bonus=args.qa_bonus)
    model.fit(docs)
    model.save(Path(args.index))
    metadata = {
        "data_dir": args.data_dir,
        "raw_dir": args.raw_dir,
        "documents": len(docs),
        "word_weight": args.word_weight,
        "char_weight": args.char_weight,
        "qa_bonus": args.qa_bonus,
        "no_train_qa": args.no_train_qa,
    }
    Path(args.index).with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    model = RagSystem.load(Path(args.index))
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

    build = subparsers.add_parser("build")
    build.add_argument("--data-dir", default="data_new")
    build.add_argument("--raw-dir", default="data/raw")
    build.add_argument("--index", default="models/rag_index.joblib")
    build.add_argument("--word-weight", type=float, default=0.58)
    build.add_argument("--char-weight", type=float, default=0.42)
    build.add_argument("--qa-bonus", type=float, default=0.08)
    build.add_argument("--no-train-qa", action="store_true")
    build.set_defaults(func=build_command)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--index", default="models/rag_index.joblib")
    predict.add_argument("--questions", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--debug")
    predict.add_argument("--top-k", type=int, default=20)
    predict.set_defaults(func=predict_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
