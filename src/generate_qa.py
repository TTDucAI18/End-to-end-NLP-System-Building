import os
import re
import json
import random
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIGURATION

RAW_DIR      = "data/raw"
TRAIN_DIR    = "data/train"
TEST_DIR     = "data/test"

TARGET_TRAIN = 1000
TARGET_TEST  = 200

# Chunking Parameters
CHUNK_SIZE   = 1500          
OVERLAP      = 200           #

# LLM Parameters
OLLAMA_HOST  = "http://localhost:11434"
GEN_MODEL    = "qwen2.5:7b"  
VAL_MODEL    = "gemma2:2b"   
TEMPERATURE  = 0.3           

MAX_WORKERS  = 3           

QUESTION_TYPES = [
    "entity-centric (hỏi về thực thể, tổ chức, nhân vật cụ thể)",
    "relationship (mối quan hệ giữa các thực thể hoặc các khoa/viện)",
    "comparison (so sánh các chương trình, chỉ tiêu hoặc mốc thời gian)",
    "definition (định nghĩa thuật ngữ, khái niệm hoặc tên gọi viết tắt)",
    "numerical (hỏi về số liệu tuyển sinh, mã ngành, năm thành lập)",
    "temporal (sự kiện gắn liền với mốc thời gian, lịch sử cụ thể)",
    "multi-hop (kết hợp từ 2 dữ kiện trở lên trong văn bản)",
    "list (yêu cầu liệt kê danh sách các điều kiện, tiêu chí hoặc nhiệm vụ)",
]

# OLLAMA CLIENT WRAPPER

class OllamaClient:
    """Client kết nối trực tiếp tới API Ollama nội bộ."""
    def __init__(self, host: str = OLLAMA_HOST):
        import requests
        self.requests = requests
        self.host = host
        try:
            r = self.requests.get(f"{host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception:
            raise ConnectionError(
                f"Không thể kết nối tới Ollama tại {host}.\n"
                "Vui lòng chắc chắn rằng Ollama đang hoạt động (ollama serve)."
            )

    def complete(self, model: str, prompt: str, temp: float = TEMPERATURE) -> str:
        """Thực hiện lệnh gọi sinh văn bản không streaming."""
        try:
            r = self.requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": 1024
                    }
                },
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            # In lỗi nhẹ để theo dõi nếu Ollama bị timeout
            # print(f"\n[WARN] API call failed: {e}")
            return ""

    def unload_model(self, model: str):
        """Yêu cầu Ollama giải phóng model khỏi VRAM để nhường chỗ cho model khác."""
        try:
            self.requests.post(
                f"{self.host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=5
            )
        except Exception:
            pass

# ADVANCED TEXT PROCESSING & CHUNKING

def load_raw_docs(raw_dir: str) -> list[dict]:
    """Đọc dữ liệu thô đã thu thập từ thư mục raw."""
    docs = []
    for p in Path(raw_dir).glob("*.txt"):
        content = p.read_text(encoding="utf-8", errors="ignore")
        m = re.match(r"SOURCE:\s*(.+)", content)
        url = m.group(1).strip() if m else str(p)
        body = re.sub(r"^(SOURCE|META):.*\n", "", content, flags=re.MULTILINE)
        body = body.lstrip("=").strip()
        if len(body) > 100:
            docs.append({"url": url, "text": body})
    return docs

def chunk_text_with_overlap(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Cắt văn bản sử dụng cửa sổ trượt gối đầu (sliding window) đảm bảo ngữ cảnh biên."""
    chunks = []
    if len(text) <= size:
        return [text.strip()]
    
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
            
        start = end - overlap
        if start >= len(text) or (end >= len(text)):
            break
            
    return chunks

# PROMPT BUILDERS (ANSWER-FIRST APPROACH)

def build_generator_prompt(chunk: str, url: str, qtypes: list[str]) -> str:
    """Prompt thiết kế theo tư duy sinh dữ liệu Answer-First tối ưu cấu trúc JSON."""
    types_str = "\n".join(f"  - {t}" for t in qtypes)
    return f"""Bạn là một chuyên gia chú giải dữ liệu cao cấp đang xây dựng bộ dữ liệu QA mẫu chuẩn phục vụ hệ thống RAG cho Đại học Quốc gia Hà Nội (VNU).

Hãy đọc kỹ đoạn văn dưới đây:
Nguồn tài liệu: {url}

<passage>
{chunk}
</passage>

Nhiệm vụ: Hãy trích xuất và sinh ra tối thiểu 2 cặp Hỏi-Đáp dựa trên kỹ thuật "Trích xuất Thực tế trước" (Answer-First) theo các quy tắc nghiêm ngặt sau:

Quy tắc thực hiện:
1. Xác định một thông tin/thực tế cụ thể có giá trị trong đoạn văn (Fact Extracted).
2. Xây dựng câu trả lời (Answer) chi tiết, đầy đủ, dựa trực tiếp và duy nhất vào thông tin thực tế đó.
3. Đặt câu hỏi (Question) tương ứng sao cho câu hỏi tự nhiên, rõ ràng, hướng tới các loại thông tin sau:
{types_str}
4. Cả câu hỏi, câu trả lời và dữ kiện trích xuất đều phải viết bằng tiếng Việt.
5. Nếu câu hỏi có nhiều câu trả lời hợp lệ từ văn bản, hãy phân cách chúng bằng dấu chấm phẩy (;).
6. Các câu hỏi và câu trả lời phải liên quan đến VNU, các khoa viện, chương trình đào tạo, chỉ tiêu tuyển sinh, sự kiện lịch sử hoặc các thông tin đặc thù của trường, không hỏi về kiến thức chung chung hoặc thông tin bên ngoài.
7. KHÔNG tự suy diễn hay bổ sung thông tin ngoài phạm vi đoạn văn được cung cấp.

Yêu cầu định dạng đầu ra bắt buộc: Bạn phải trả về định dạng mảng JSON thuần túy, tuyệt đối KHÔNG viết thêm bất kỳ câu giải thích nào ở đầu hay cuối phản hồi của bạn. Ví dụ định dạng:
[
  {{
    "fact_extracted": "Dữ kiện thực tế trích xuất từ văn bản",
    "answer": "Câu trả lời chi tiết chính xác",
    "question": "Câu hỏi tương ứng"
  }}
]"""

def build_verifier_prompt(chunk: str, question: str, answer: str) -> str:
    """Prompt dành riêng cho mô hình Verifier đánh giá tính chính xác."""
    return f"""Bạn là một chuyên gia kiểm duyệt dữ liệu khắt khe. Hãy thẩm định xem cặp Câu hỏi - Câu trả lời dưới đây có hoàn toàn chính xác và được hỗ trợ trực tiếp từ đoạn văn bản (Passage) không.

<passage>
{chunk}
</passage>

Câu hỏi: {question}
Câu trả lời: {answer}

Yêu cầu:
- Nếu toàn bộ thông tin trong câu trả lời được xác thực trực tiếp và không có yếu tố suy diễn hay ảo tưởng ngoài đoạn văn, hãy trả về: YES
- Nếu có bất kỳ chi tiết nào sai lệch, suy diễn quá đà, không có trong văn bản, hoặc câu hỏi không rõ nghĩa, hãy trả về: NO

Chỉ phản hồi đúng 1 từ duy nhất: YES hoặc NO. Không giải thích thêm."""

# HARD NEGATIVE MINING

def calculate_token_overlap(str1: str, str2: str) -> float:
    words1 = set(re.findall(r'\w+', str1.lower()))
    words2 = set(re.findall(r'\w+', str2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

def mine_hard_negative(question: str, pos_chunk_idx: int, all_chunks: list[str]) -> str:
    best_score = -1.0
    hard_negative = ""
    for idx, chunk in enumerate(all_chunks):
        if idx == pos_chunk_idx:
            continue
        score = calculate_token_overlap(question, chunk)
        if score > best_score:
            best_score = score
            hard_negative = chunk
    return hard_negative if best_score > 0.05 else ""

# PARSING & DEDUPLICATION

def parse_json_response(raw_text: str) -> list[dict]:
    """
    Bộ phân tích cú pháp JSON cải tiến cực kỳ mạnh mẽ chống lỗi định dạng của LLM.
    Tự động lọc các ký tự thừa, khối mã markdown, hoặc trích xuất thủ công từng cặp JSON.
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    # 1. Thử bóc tách trực tiếp khối mảng [] chuẩn nếu mô hình trả về đúng
    try:
        clean = re.sub(r"^```[a-z]*\n?", "", raw_text.strip(), flags=re.IGNORECASE).rstrip("`").strip()
        match = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                # Chỉ lấy các object có đủ question/answer
                return [p for p in parsed if isinstance(p, dict) and "question" in p and "answer" in p]
    except Exception:
        pass

    results = []
    stack = []
    start_idx = -1
    
    for i, char in enumerate(raw_text):
        if char == '{':
            if len(stack) == 0:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if len(stack) > 0:
                stack.pop()
                if len(stack) == 0 and start_idx != -1:
                    block = raw_text[start_idx:i+1]
                    try:
                        block = re.sub(r',\s*\}', '}', block)
                        obj = json.loads(block)
                        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
    return results

def deduplicate_qa(pairs: list[dict]) -> list[dict]:
    seen = set()
    unique_pairs = []
    for p in pairs:
        norm_key = re.sub(r"\W+", " ", str(p.get("question", "")).lower()).strip()
        if norm_key and norm_key not in seen:
            seen.add(norm_key)
            unique_pairs.append(p)
    return unique_pairs

# PIPELINE EXECUTION

def main():
    parser = argparse.ArgumentParser(description="VNU Optimized QA Pipeline (Speedup Edition)")
    parser.add_argument("--gen_model", default=GEN_MODEL, help="Mô hình sinh Generator")
    parser.add_argument("--val_model", default=VAL_MODEL, help="Mô hình kiểm duyệt Verifier")
    parser.add_argument("--train_size", type=int, default=TARGET_TRAIN, help="Số lượng train tối đa")
    parser.add_argument("--test_size", type=int, default=TARGET_TEST, help="Số lượng test tối đa")
    args = parser.parse_args()

    print("=" * 60)
    print("VNU HIGH-SPEED QA PIPELINE — OLLAMA (BUG FIXED)")
    print("=" * 60)

    try:
        client = OllamaClient()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return

    docs = load_raw_docs(RAW_DIR)
    if not docs:
        print(f"[ERROR] Không tìm thấy dữ liệu thô tại {RAW_DIR}/")
        return
    
    print(f"Đã tải {len(docs)} tài liệu từ {RAW_DIR}/.")
    
    flat_chunks = []
    for doc_idx, doc in enumerate(docs):
        chunks = chunk_text_with_overlap(doc["text"])
        for chunk_idx, chunk in enumerate(chunks):
            flat_chunks.append({
                "doc_idx": doc_idx,
                "doc_url": doc["url"],
                "chunk_idx": chunk_idx,
                "chunk_text": chunk,
                "all_doc_chunks": chunks
            })

    total_target = args.train_size + args.test_size
    print(f"Tổng số chunks: {len(flat_chunks)}. Cần khoảng {total_target} cặp QA.")

    # PHASE 1: GENERATION
    print(f"\n[PHASE 1/2] Đang chạy sinh câu hỏi thô bằng {args.gen_model}...")
    raw_qa_pool = []

    def process_chunk_generation(item):
        sampled_qtypes = random.sample(QUESTION_TYPES, k=min(3, len(QUESTION_TYPES)))
        prompt = build_generator_prompt(item["chunk_text"], item["doc_url"], sampled_qtypes)
        
        response_text = client.complete(args.gen_model, prompt, temp=TEMPERATURE)
        pairs = parse_json_response(response_text)
        
        results = []
        for pair in pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if len(q) >= 15 and len(a) >= 5:
                results.append({
                    "question": q,
                    "answer": a,
                    "source_chunk": item["chunk_text"],
                    "chunk_idx": item["chunk_idx"],
                    "all_chunks": item["all_doc_chunks"],
                    "url": item["doc_url"],
                    "doc_idx": item["doc_idx"]
                })
        return results

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk_generation, item): item for item in flat_chunks}
        pbar_gen = tqdm(total=len(flat_chunks), desc="Sinh QA thô", unit="chunk")
        
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                if chunk_results:
                    raw_qa_pool.extend(chunk_results)
            except Exception:
                pass
            pbar_gen.update(1)
            
            if len(raw_qa_pool) >= total_target * 1.8:
                executor.shutdown(wait=False, cancel_futures=True)
                break
        pbar_gen.close()

    print(f"-> Đang dọn dẹp {args.gen_model} khỏi VRAM...")
    client.unload_model(args.gen_model)
    time.sleep(2.0)

    # PHASE 2: VERIFICATION
    print(f"\n[PHASE 2/2] Đang thẩm định chéo bằng {args.val_model}...")
    verified_data = []

    raw_qa_pool = deduplicate_qa(raw_qa_pool)
    print(f"Tổng số cặp QA cần thẩm định: {len(raw_qa_pool)}")

    if len(raw_qa_pool) == 0:
        print("\n[ERROR] Không tạo được cặp dữ liệu nào ở Phase 1. Hãy kiểm tra lại LLM Prompt hoặc thử đổi model mạnh hơn.")
        return

    def process_verification(qa_item):
        val_prompt = build_verifier_prompt(qa_item["source_chunk"], qa_item["question"], qa_item["answer"])
        verification = client.complete(args.val_model, val_prompt, temp=0.0).strip().upper()
        
        if "YES" in verification:
            hard_neg = mine_hard_negative(qa_item["question"], qa_item["chunk_idx"], qa_item["all_chunks"])
            return {
                "question": qa_item["question"],
                "answer": qa_item["answer"],
                "source_chunk": qa_item["source_chunk"],
                "hard_negative_chunk": hard_neg,
                "url": qa_item["url"],
                "chunk_id": f"doc_{qa_item['doc_idx']}_chk_{qa_item['chunk_idx']}"
            }
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_verification, qa_item): qa_item for qa_item in raw_qa_pool}
        pbar_val = tqdm(total=len(raw_qa_pool), desc="Thẩm định", unit="cặp")
        
        for future in as_completed(futures):
            try:
                verified_item = future.result()
                if verified_item:
                    verified_data.append(verified_item)
            except Exception:
                pass
            pbar_val.update(1)
            
            if len(verified_data) >= total_target:
                executor.shutdown(wait=False, cancel_futures=True)
                break
        pbar_val.close()

    client.unload_model(args.val_model)

    # EXPORT DATA
    unique_data = deduplicate_qa(verified_data)
    random.shuffle(unique_data)
    total_count = len(unique_data)

    if total_count == 0:
        print("\n[ERROR] Không tạo được cặp dữ liệu sạch nào vượt qua kiểm định.")
        return

    n_test = min(args.test_size, int(total_count * 0.15))
    n_train = total_count - n_test

    test_split = unique_data[:n_test]
    train_split = unique_data[n_test:n_test + n_train]

    def export_split(split_data, directory):
        os.makedirs(directory, exist_ok=True)
        jsonl_path = os.path.join(directory, "dataset.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        q_path = os.path.join(directory, "questions.txt")
        a_path = os.path.join(directory, "reference_answers.txt")
        with open(q_path, "w", encoding="utf-8") as qf, open(a_path, "w", encoding="utf-8") as af:
            for item in split_data:
                qf.write(item["question"] + "\n")
                af.write(item["answer"] + "\n")

    print("\nĐang xuất kết quả...")
    export_split(test_split, TEST_DIR)
    export_split(train_split, TRAIN_DIR)

    print(f"\n{'='*60}")
    print(f"Tổng số cặp QA sạch thu thập: {total_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()