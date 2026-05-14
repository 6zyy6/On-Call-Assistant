import math
import re
from contextlib import asynccontextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from threading import RLock
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+|[&]|[\u4e00-\u9fff]+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unescape(text)).strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    tokens: List[str] = []

    for match in TOKEN_PATTERN.finditer(normalized):
        chunk = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
                continue
            for size in (2, 3):
                if len(chunk) >= size:
                    tokens.extend(chunk[index:index + size] for index in range(len(chunk) - size + 1))
        else:
            tokens.append(chunk)

    return tokens


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._text_parts: List[str] = []
        self._title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return

        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    @property
    def visible_text(self) -> str:
        return normalize_text(" ".join(self._text_parts))

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self._title_parts))


def extract_visible_text_and_title(html: str) -> tuple[str, str]:
    parser = VisibleTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.visible_text, parser.title


def build_snippet(text: str, query: str, radius: int = 50) -> str:
    normalized_query = normalize_text(query)
    if not text:
        return ""

    if normalized_query:
        lower_text = text.lower()
        lower_query = normalized_query.lower()
        position = lower_text.find(lower_query)
        if position >= 0:
            start = max(0, position - radius)
            end = min(len(text), position + len(normalized_query) + radius)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet

    return text[: radius * 2] + ("..." if len(text) > radius * 2 else "")


@dataclass
class StoredDocument:
    id: str
    title: str
    html: str
    text: str
    tokens: List[str]
    term_freq: Counter
    length: int


class DocumentStore:
    def __init__(self) -> None:
        self._documents: Dict[str, StoredDocument] = {}
        self._doc_freq: defaultdict[str, int] = defaultdict(int)
        self._total_terms = 0
        self._lock = RLock()

    def upsert(self, doc_id: str, html: str) -> StoredDocument:
        text, title = extract_visible_text_and_title(html)
        if not text:
            raise HTTPException(status_code=400, detail="Document has no visible text")

        tokens = tokenize(text)
        if not tokens:
            raise HTTPException(status_code=400, detail="Document has no indexable text")

        document = StoredDocument(
            id=doc_id,
            title=title or doc_id,
            html=html,
            text=text,
            tokens=tokens,
            term_freq=Counter(tokens),
            length=len(tokens),
        )

        with self._lock:
            previous = self._documents.get(doc_id)
            if previous:
                self._remove_from_index(previous)
            self._documents[doc_id] = document
            self._add_to_index(document)

        return document

    def _remove_from_index(self, document: StoredDocument) -> None:
        self._total_terms -= document.length
        for token in document.term_freq:
            self._doc_freq[token] -= 1
            if self._doc_freq[token] <= 0:
                del self._doc_freq[token]

    def _add_to_index(self, document: StoredDocument) -> None:
        self._total_terms += document.length
        for token in document.term_freq:
            self._doc_freq[token] += 1

    def search(self, query: str, limit: int = 10) -> List[dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            documents = list(self._documents.values())
            doc_count = len(documents)
            if doc_count == 0:
                return []

            average_length = self._total_terms / doc_count if self._total_terms else 0.0
            results = []
            for document in documents:
                score = self._bm25_score(document, query_tokens, doc_count, average_length)
                if score <= 0:
                    continue
                results.append(
                    {
                        "id": document.id,
                        "title": document.title,
                        "snippet": build_snippet(document.text, query),
                        "score": round(score, 6),
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _bm25_score(
        self,
        document: StoredDocument,
        query_tokens: List[str],
        doc_count: int,
        average_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        score = 0.0
        denominator_base = k1 * (1 - b + b * document.length / average_length) if average_length else k1

        for token in query_tokens:
            frequency = document.term_freq.get(token, 0)
            if frequency == 0:
                continue

            doc_frequency = self._doc_freq.get(token, 0)
            idf = math.log(1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            score += idf * ((frequency * (k1 + 1)) / (frequency + denominator_base))

        return score


class DocumentPayload(BaseModel):
    id: str
    html: str


DATA_DIR = Path(__file__).parent / "data"
store = DocumentStore()

def load_seed_documents() -> None:
    if not DATA_DIR.exists():
        return

    for html_file in sorted(DATA_DIR.glob("*.html")):
        store.upsert(html_file.stem, html_file.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_seed_documents()
    yield


app = FastAPI(
    title="On-Call Assistant Search Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/documents")
def upsert_document(payload: DocumentPayload) -> dict:
    document = store.upsert(payload.id, payload.html)
    return {"id": document.id, "title": document.title}


@app.get("/v1/search")
def search_documents(q: str = Query(..., min_length=1)) -> dict:
    return {"query": q, "results": store.search(q)}


@app.get("/v1", response_class=HTMLResponse)
def search_page() -> str:
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>On-Call 搜索</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, sans-serif;
    }
    body {
      margin: 0;
      background: #f5f7fb;
      color: #1f2937;
    }
    main {
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    h1 {
      margin: 0 0 20px;
      font-size: 28px;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      margin-bottom: 24px;
    }
    input, button {
      height: 44px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 16px;
      padding: 0 14px;
    }
    button {
      background: #2563eb;
      border-color: #2563eb;
      color: white;
      cursor: pointer;
    }
    ul {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 12px;
    }
    li {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
    }
    .meta {
      color: #64748b;
      font-size: 14px;
      margin-top: 8px;
    }
    .empty {
      color: #64748b;
      padding: 12px 0;
    }
  </style>
</head>
<body>
  <main>
    <h1>On-Call SOP 搜索</h1>
    <form id="search-form">
      <input id="query" name="q" type="text" placeholder="输入关键词，例如 OOM / 故障 / CDN / &" autocomplete="off">
      <button type="submit">搜索</button>
    </form>
    <div id="status" class="empty">请输入关键词开始搜索。</div>
    <ul id="results"></ul>
  </main>
  <script>
    const form = document.getElementById("search-form");
    const input = document.getElementById("query");
    const results = document.getElementById("results");
    const status = document.getElementById("status");

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = input.value.trim();
      results.innerHTML = "";

      if (!query) {
        status.textContent = "请输入关键词开始搜索。";
        return;
      }

      status.textContent = "搜索中...";
      const response = await fetch(`/v1/search?q=${encodeURIComponent(query)}`);
      const payload = await response.json();

      if (!payload.results.length) {
        status.textContent = `没有找到与 "${query}" 相关的文档。`;
        return;
      }

      status.textContent = `找到 ${payload.results.length} 条结果。`;
      results.innerHTML = payload.results.map((item) => `
        <li>
          <strong>${escapeHtml(item.title)}</strong>
          <div class="meta">ID: ${escapeHtml(item.id)} | Score: ${item.score}</div>
          <p>${escapeHtml(item.snippet)}</p>
        </li>
      `).join("");
    });
  </script>
</body>
</html>
"""


@app.get("/")
def root() -> dict:
    return {"message": "Visit /v1 for the search UI."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
