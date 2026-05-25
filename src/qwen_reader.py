import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.raw_rag import RawRag, load_lines


def build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))
    return (
        "Bạn là hệ thống hỏi đáp factual QA về VNU/UET.\n"
        "Chỉ dùng CONTEXT để trả lời. Nếu không đủ thông tin, hãy trả lời ngắn nhất có thể từ context liên quan nhất.\n"
        "Trả lời bằng một cụm ngắn, không giải thích, không thêm câu dẫn.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.split(r"\n|</s>|<\|im_end\|>", text)[0].strip()
    text = re.sub(r"^(ANSWER|Đáp án|Trả lời)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text.split()) > 30:
        text = " ".join(text.split()[:30])
    return text.strip(" .,:;")


def load_qwen(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=3072)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return clean_answer(tokenizer.decode(generated, skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-model", default="models/raw_rag_distinct.joblib")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", default="reports/qwen_reader_debug.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    rag = RawRag.load(Path(args.rag_model))
    tokenizer, model = load_qwen(args.qwen_model)
    questions = load_lines(Path(args.questions))
    outputs = []

    Path(args.debug).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.debug).open("w", encoding="utf-8") as debug:
        for question in questions:
            retrieved = rag.retrieve(question, top_k=args.top_k)
            contexts = [rag.docs[idx].text for idx, _ in retrieved]
            prompt = build_prompt(question, contexts)
            answer = generate_answer(tokenizer, model, prompt, args.max_new_tokens)
            if not answer:
                answer = rag.answer(question, top_k=20)["answer"]
            outputs.append(answer)
            debug.write(
                json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": [
                            {
                                "doc_id": rag.docs[idx].doc_id,
                                "source": rag.docs[idx].source,
                                "kind": rag.docs[idx].kind,
                                "score": score,
                            }
                            for idx, score in retrieved
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(outputs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
