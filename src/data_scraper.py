import os
import re
import time
import hashlib
import requests
import pdfplumber
from io import BytesIO
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm

# CONFIG

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; VNU-NLP-Assignment-Scraper/1.0; "
        "research use only)"
    )
}

REQUEST_DELAY = 1.5          
MAX_DEPTH     = 2           
MAX_PAGES_PER_SITE = 50    

# SEED URLs  

SEED_URLS = [
    # ── VNU Main ──────────────────────────────
    "https://vnu.edu.vn/home/",    

    # ── Wikipedia ─────────────────────────────
    "https://vi.wikipedia.org/wiki/%C4%90%E1%BA%A1i_h%E1%BB%8Dc_Qu%E1%BB%91c_gia_H%C3%A0_N%E1%BB%99i",

    # ── UET (University of Engineering & Technology) ──
    "https://uet.vnu.edu.vn/",
    "https://uet.vnu.edu.vn/gioi-thieu/",
    "https://uet.vnu.edu.vn/dao-tao/",
    "https://tuyensinh.uet.vnu.edu.vn/",

    # ── VNU Admissions Portal ─────────────────
    "https://tuyensinh.vnu.edu.vn/",
    "https://tuyensinh.vnu.edu.vn/thong-tin-tuyen-sinh",
]

# Sub-pages to try appending to each member-university base URL
SUBPAGE_SUFFIXES = [
    "/gioi-thieu/", "/dao-tao/", "/tuyen-sinh/",
    "/lich-su/",    "/to-chuc/", "/tin-tuc/",
    "/about/",      "/history/", "/admissions/",
    "/schedule/",   "/events/",  "/vendors/",
    "/upcoming-events/",
]

# HELPERS

def url_to_filename(url: str) -> str:
    """Deterministic safe filename from URL."""
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", urlparse(url).path.lower()).strip("_")
    return f"{slug[:60]}_{h}.txt"


def clean_text(text: str) -> str:
    """Remove excessive whitespace and boilerplate artifacts."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_same_domain(url: str, base: str) -> bool:
    """True if url belongs to the same domain (or subdomain) as base."""
    u = urlparse(url)
    b = urlparse(base)
    # allow subdomains: uet.vnu.edu.vn matches vnu.edu.vn
    return u.netloc.endswith(b.netloc) or b.netloc.endswith(u.netloc)


def fetch(url: str, timeout: int = 15):
    """GET with error handling. Returns Response or None."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"  [SKIP] {url} — {e}")
        return None


# HTML SCRAPER

def extract_text_from_html(html: str, url: str) -> tuple[str, list[str]]:
    """
    Returns (clean_text, list_of_child_links).
    Strips nav/footer/script/style boilerplate.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Try to get main content area first
    main = (soup.find("main") or
            soup.find("div", {"id": re.compile(r"content|main|body", re.I)}) or
            soup.find("div", {"class": re.compile(r"content|main|article", re.I)}) or
            soup.body)

    raw = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    text = clean_text(raw)

    # Collect internal links for crawling
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        parsed = urlparse(href)
        if parsed.scheme in ("http", "https") and is_same_domain(href, url):
            # Drop anchors and query strings to de-duplicate
            clean_href = parsed._replace(fragment="", query="").geturl()
            links.append(clean_href)

    return text, links


# PDF SCRAPER

def extract_text_from_pdf(content: bytes, url: str) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    pages_text = []
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    pages_text.append(f"[Page {i+1}]\n{t}")
    except Exception as e:
        print(f"  [PDF ERROR] {url} — {e}")
    return clean_text("\n\n".join(pages_text))


# CRAWLER

def save_doc(url: str, text: str, meta: str = ""):
    """Write a document to data/raw/."""
    if len(text) < 100:          # skip near-empty pages
        return
    fname = url_to_filename(url)
    path  = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {url}\n")
        if meta:
            f.write(f"META: {meta}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    return path


def crawl(seed_url: str, max_depth: int = MAX_DEPTH,
          max_pages: int = MAX_PAGES_PER_SITE) -> int:
    """BFS crawler starting from seed_url. Returns number of pages saved."""
    visited = set()
    queue   = [(seed_url, 0)]          # (url, depth)
    saved   = 0

    while queue and saved < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        time.sleep(REQUEST_DELAY)
        r = fetch(url)
        if r is None:
            continue

        content_type = r.headers.get("Content-Type", "")

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = extract_text_from_pdf(r.content, url)
            links = []
        elif "html" in content_type:
            text, links = extract_text_from_html(r.text, url)
            # Queue child links up to max_depth
            if depth < max_depth:
                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))
        else:
            continue  # skip binary / unknown types

        if save_doc(url, text):
            saved += 1
            print(f"  ✓ [{saved:03d}] depth={depth}  {url[:80]}")

    return saved


# MAIN

def main():
    print("=" * 60)
    print("VNU Knowledge Base Scraper")
    print("=" * 60)

    total = 0

    # 1. Crawl primary seed URLs
    for seed in tqdm(SEED_URLS, desc="Seed URLs"):
        print(f"\n→ Crawling: {seed}")
        n = crawl(seed)
        total += n
        print(f"  Saved {n} pages from this seed.")

    # 2. Try common sub-page suffixes for member universities
    MEMBER_BASES = [
        "https://uet.vnu.edu.vn",
        "https://hus.vnu.edu.vn",
        "https://ulis.vnu.edu.vn",
        "https://ussh.vnu.edu.vn",
        "https://vnu.edu.vn",
    ]
    print("\n→ Probing sub-pages on member universities…")
    for base in MEMBER_BASES:
        for suffix in SUBPAGE_SUFFIXES:
            url = base + suffix
            time.sleep(REQUEST_DELAY)
            r = fetch(url)
            if r is None:
                continue
            content_type = r.headers.get("Content-Type", "")
            if "html" in content_type:
                text, _ = extract_text_from_html(r.text, url)
                if save_doc(url, text):
                    total += 1
                    print(f"  ✓ sub-page: {url}")

    print(f"\n{'='*60}")
    print(f"Done! Total documents saved: {total}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")

    # 3. Create placeholder files for annotation
    for split in ("train", "test"):
        for fname in ("questions.txt", "reference_answers.txt"):
            path = f"data/{split}/{fname}"
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# Add your {split} {fname.replace('.txt','')} here, one per line.\n")
    print("\nPlaceholder annotation files created in data/train/ and data/test/")


if __name__ == "__main__":
    main()