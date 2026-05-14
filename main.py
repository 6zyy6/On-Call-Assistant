import math
import json
import os
import re
from contextlib import asynccontextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+|[&]|[\u4e00-\u9fff]+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
BLOCK_PATTERN = re.compile(
    r"<(h[1-3]|p|li)[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
IGNORED_BLOCK_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

QUERY_EXPANSIONS = {
    "服务器挂了": ["服务不可用", "服务宕机", "服务崩溃", "服务超时", "后端故障", "基础设施故障", "后端服务", "SRE基础设施", "Kubernetes", "Ingress"],
    "挂了": ["不可用", "宕机", "崩溃", "中断", "超时"],
    "宕机": ["不可用", "中断", "崩溃", "故障"],
    "黑客攻击": ["安全攻击", "入侵", "SQL注入", "DDoS", "恶意流量", "安全事件", "信息安全", "WAF", "漏洞利用"],
    "攻击": ["入侵", "DDoS", "注入", "恶意请求"],
    "机器学习模型出问题": ["模型服务故障", "模型推理异常", "特征服务故障", "GPU故障", "效果下降", "AI算法", "推荐服务", "搜索服务"],
    "模型出问题": ["模型服务故障", "推理异常", "特征异常", "GPU故障", "AI算法"],
    "模型": ["推理", "特征服务", "效果", "GPU", "AI算法"],
}

DOMAIN_HINTS = {
    "后端": ["后端", "服务", "接口", "网关", "依赖", "超时", "熔断"],
    "sre": ["sre", "基础设施", "kubernetes", "k8s", "ingress", "etcd", "控制平面", "网关", "云资源"],
    "数据库": ["数据库", "dba", "mysql", "redis", "postgresql", "主从", "连接池"],
    "安全": ["安全", "攻击", "注入", "ddos", "漏洞", "风控", "恶意"],
    "ai": ["ai", "算法", "模型", "推理", "特征", "gpu", "实验"],
    "数据": ["数据", "hadoop", "flink", "kafka", "hdfs", "离线", "实时"],
}

QUERY_INTENTS = [
    {"triggers": ["服务器", "服务", "挂了", "宕机", "不可用", "中断", "超时"], "domains": ["后端", "sre"]},
    {"triggers": ["黑客", "攻击", "入侵", "注入", "ddos", "恶意"], "domains": ["安全"]},
    {"triggers": ["模型", "机器学习", "推理", "特征", "gpu", "实验"], "domains": ["ai"]},
    {"triggers": ["数据库", "主从", "连接池", "慢查询", "binlog"], "domains": ["数据库"]},
    {"triggers": ["hadoop", "flink", "kafka", "hdfs", "离线任务", "实时计算"], "domains": ["数据"]},
]


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unescape(text)).strip()


def strip_tags(html_fragment: str) -> str:
    cleaned = IGNORED_BLOCK_PATTERN.sub(" ", html_fragment)
    cleaned = TAG_PATTERN.sub(" ", cleaned)
    return normalize_text(cleaned)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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

    for candidate in [normalized_query] + expand_query(query):
        if not candidate:
            continue
        lower_text = text.lower()
        lower_query = candidate.lower()
        position = lower_text.find(lower_query)
        if position >= 0:
            start = max(0, position - radius)
            end = min(len(text), position + len(candidate) + radius)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet

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


def expand_query(query: str) -> List[str]:
    normalized = normalize_text(query)
    if not normalized:
        return []

    expansions: List[str] = []
    seen = {normalized}

    for phrase, related in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            for item in related:
                candidate = normalize_text(item)
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    expansions.append(candidate)

    if any(token in normalized for token in ("服务器", "服务")):
        for candidate in ("服务故障", "服务不可用", "后端服务", "基础设施"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    if "挂" in normalized or "宕机" in normalized:
        for candidate in ("故障", "不可用", "中断", "超时"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    return expansions


def build_hybrid_query(query: str) -> str:
    parts = [normalize_text(query)] + expand_query(query)
    return " ".join(part for part in parts if part)


@dataclass
class StoredDocument:
    id: str
    title: str
    html: str
    text: str
    tokens: List[str]
    term_freq: Counter
    length: int


@dataclass
class StoredChunk:
    id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    context_text: str
    tokens: List[str]
    term_freq: Counter
    length: int


def extract_document_chunks(html: str, title: str, doc_id: str) -> List[StoredChunk]:
    cleaned_html = IGNORED_BLOCK_PATTERN.sub(" ", html)
    headings = {"h1": title, "h2": "", "h3": ""}
    chunks: List[StoredChunk] = []
    chunk_index = 0

    for match in BLOCK_PATTERN.finditer(cleaned_html):
        tag = match.group(1).lower()
        content = strip_tags(match.group(2))
        if not content:
            continue

        if tag in {"h1", "h2", "h3"}:
            headings[tag] = content
            if tag == "h1":
                headings["h2"] = ""
                headings["h3"] = ""
            elif tag == "h2":
                headings["h3"] = ""
            continue

        heading_parts = [headings["h1"], headings["h2"], headings["h3"]]
        heading = " | ".join(part for part in heading_parts if part)
        context_text = normalize_text(f"{title} {heading} {content}")
        tokens = tokenize(context_text)
        if not tokens:
            continue

        chunks.append(
            StoredChunk(
                id=f"{doc_id}::chunk-{chunk_index}",
                doc_id=doc_id,
                doc_title=title,
                heading=heading,
                text=content,
                context_text=context_text,
                tokens=tokens,
                term_freq=Counter(tokens),
                length=len(tokens),
            )
        )
        chunk_index += 1

    if chunks:
        return chunks

    fallback_text = strip_tags(html)
    tokens = tokenize(fallback_text)
    if not tokens:
        return []
    return [
        StoredChunk(
            id=f"{doc_id}::chunk-0",
            doc_id=doc_id,
            doc_title=title,
            heading=title,
            text=fallback_text,
            context_text=normalize_text(f"{title} {fallback_text}"),
            tokens=tokens,
            term_freq=Counter(tokens),
            length=len(tokens),
        )
    ]


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

    def list_documents(self) -> List[StoredDocument]:
        with self._lock:
            return list(self._documents.values())

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


class SemanticSearchEngine:
    def __init__(self) -> None:
        self._faiss = None
        self._np = None
        self._documents: List[StoredDocument] = []
        self._chunks: List[StoredChunk] = []
        self._index = None
        self._last_error: Optional[str] = None
        self._lock = RLock()

    def _ensure_backend(self) -> None:
        if self._faiss is not None and self._np is not None:
            return

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            self._last_error = "Semantic search dependencies are missing. Install faiss-cpu."
            raise RuntimeError(self._last_error) from exc

        self._faiss = faiss
        self._np = np

    def _api_key(self) -> str:
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            self._last_error = "Missing DASHSCOPE_API_KEY. Set it before using /v2 semantic search."
            raise RuntimeError(self._last_error)
        return api_key

    def _base_url(self) -> str:
        return os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")

    def _model_name(self) -> str:
        return os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4").strip() or "text-embedding-v4"

    def _dimensions(self) -> int:
        raw_value = os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024").strip()
        try:
            return int(raw_value)
        except ValueError as exc:
            self._last_error = "DASHSCOPE_EMBEDDING_DIMENSIONS must be an integer."
            raise RuntimeError(self._last_error) from exc

    def _embed_texts(self, texts: List[str]) -> "object":
        self._ensure_backend()
        api_key = self._api_key()
        endpoint = f"{self._base_url()}/services/embeddings/text-embedding/text-embedding"
        model_name = self._model_name()
        dimensions = self._dimensions()
        all_vectors = []

        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            payload = {
                "model": model_name,
                "input": {"texts": batch},
                "parameters": {
                    "dimension": dimensions,
                    "output_type": "dense",
                },
            }
            request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urlopen(request, timeout=60) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                self._last_error = f"DashScope embedding request failed: HTTP {exc.code}. {details}".strip()
                raise RuntimeError(self._last_error) from exc
            except URLError as exc:
                self._last_error = f"DashScope embedding request failed: {exc.reason}"
                raise RuntimeError(self._last_error) from exc

            embeddings = response_payload.get("output", {}).get("embeddings", [])
            if len(embeddings) != len(batch):
                self._last_error = "DashScope returned an unexpected embedding payload."
                raise RuntimeError(self._last_error)

            ordered = sorted(embeddings, key=lambda item: item.get("text_index", 0))
            all_vectors.extend(item["embedding"] for item in ordered)

        vectors = self._np.asarray(all_vectors, dtype="float32")
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def rebuild(self, documents: List[StoredDocument]) -> None:
        with self._lock:
            self._documents = list(documents)
            self._chunks = []
            for document in self._documents:
                self._chunks.extend(extract_document_chunks(document.html, document.title, document.id))

            if not self._chunks:
                self._index = None
                self._last_error = None
                return

            vectors = self._embed_texts([chunk.context_text for chunk in self._chunks])
            self._last_error = None
            index = self._faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._index = index

    def search(self, query: str, limit: int = 10) -> List[dict]:
        query = normalize_text(query)
        if not query:
            return []

        with self._lock:
            if self._last_error and self._index is None:
                raise RuntimeError(self._last_error)
            if self._index is None or not self._chunks:
                return []

            expanded_terms = expand_query(query)
            hybrid_query = build_hybrid_query(query)
            query_vector = self._embed_texts([hybrid_query])
            top_k = min(max(limit * 6, 12), len(self._chunks))
            semantic_scores, semantic_indices = self._index.search(query_vector, top_k)
            bm25_scores = self._score_chunks_bm25(hybrid_query)
            keyword_scores = self._score_heading_matches(query, expanded_terms)
            intent_scores = self._score_intent_boosts(query, expanded_terms)

            semantic_map = {
                int(index): float(score)
                for score, index in zip(semantic_scores[0], semantic_indices[0])
                if index >= 0
            }
            candidate_indices = set(semantic_map.keys())
            candidate_indices.update(
                index
                for index, _ in sorted(
                    enumerate(bm25_scores),
                    key=lambda item: item[1],
                    reverse=True,
                )[:top_k]
                if _ > 0
            )

            max_semantic = max(semantic_map.values(), default=1.0)
            max_bm25 = max((bm25_scores[index] for index in candidate_indices), default=1.0)

            doc_results: Dict[str, dict] = {}
            for chunk_index in candidate_indices:
                chunk = self._chunks[chunk_index]
                semantic_score = semantic_map.get(chunk_index, 0.0)
                bm25_score = bm25_scores[chunk_index]
                heading_boost = keyword_scores[chunk_index]
                intent_boost = intent_scores[chunk_index]
                normalized_semantic = semantic_score / max_semantic if max_semantic > 0 else 0.0
                normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
                final_score = (
                    0.45 * normalized_semantic
                    + 0.25 * normalized_bm25
                    + 0.15 * heading_boost
                    + 0.15 * intent_boost
                )

                current = doc_results.get(chunk.doc_id)
                candidate = {
                    "id": chunk.doc_id,
                    "title": chunk.doc_title,
                    "snippet": build_snippet(chunk.context_text, query),
                    "score": round(final_score, 6),
                    "_raw_score": final_score,
                }
                if current is None or candidate["_raw_score"] > current["_raw_score"]:
                    doc_results[chunk.doc_id] = candidate

            results = sorted(doc_results.values(), key=lambda item: item["_raw_score"], reverse=True)[:limit]
            for item in results:
                item.pop("_raw_score", None)
            return results

    def _score_chunks_bm25(self, query: str) -> List[float]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0 for _ in self._chunks]

        doc_freq: defaultdict[str, int] = defaultdict(int)
        total_length = 0
        for chunk in self._chunks:
            total_length += chunk.length
            for token in chunk.term_freq:
                doc_freq[token] += 1

        doc_count = len(self._chunks)
        average_length = total_length / doc_count if doc_count else 0.0
        scores: List[float] = []
        for chunk in self._chunks:
            score = 0.0
            denominator_base = 1.5 * (1 - 0.75 + 0.75 * chunk.length / average_length) if average_length else 1.5
            for token in query_tokens:
                frequency = chunk.term_freq.get(token, 0)
                if frequency == 0:
                    continue
                token_doc_freq = doc_freq.get(token, 0)
                idf = math.log(1 + (doc_count - token_doc_freq + 0.5) / (token_doc_freq + 0.5))
                score += idf * ((frequency * 2.5) / (frequency + denominator_base))
            scores.append(score)
        return scores

    def _score_heading_matches(self, query: str, expanded_terms: List[str]) -> List[float]:
        phrases = [normalize_text(query)] + expanded_terms
        tokens = tokenize(" ".join(phrases))
        phrase_set = {phrase.lower() for phrase in phrases if phrase}
        token_set = {token.lower() for token in tokens if token}
        scores: List[float] = []

        for chunk in self._chunks:
            heading_text = normalize_text(f"{chunk.doc_title} {chunk.heading}").lower()
            score = 0.0

            for phrase in phrase_set:
                if phrase and phrase in heading_text:
                    score += 0.35

            for token in token_set:
                if token and token in heading_text:
                    score += 0.08

            for domain_terms in DOMAIN_HINTS.values():
                match_count = sum(1 for term in domain_terms if term.lower() in heading_text and term.lower() in " ".join(phrase_set).lower())
                if match_count:
                    score += min(0.2, 0.05 * match_count)

            scores.append(min(score, 1.0))

        return scores

    def _score_intent_boosts(self, query: str, expanded_terms: List[str]) -> List[float]:
        normalized_query = normalize_text(query).lower()
        normalized_expansions = [term.lower() for term in expanded_terms]
        combined_query = " ".join([normalized_query] + normalized_expansions)
        target_domains: List[str] = []

        for intent in QUERY_INTENTS:
            if any(trigger.lower() in combined_query for trigger in intent["triggers"]):
                for domain in intent["domains"]:
                    if domain not in target_domains:
                        target_domains.append(domain)

        if not target_domains:
            return [0.0 for _ in self._chunks]

        scores: List[float] = []
        for chunk in self._chunks:
            title_text = normalize_text(chunk.doc_title).lower()
            heading_text = normalize_text(f"{chunk.doc_title} {chunk.heading}").lower()
            body_text = chunk.context_text.lower()
            score = 0.0

            for domain in target_domains:
                hints = DOMAIN_HINTS.get(domain, [])
                title_hits = sum(1 for hint in hints if hint.lower() in title_text)
                heading_hits = sum(1 for hint in hints if hint.lower() in heading_text)
                body_hits = sum(1 for hint in hints if hint.lower() in body_text)
                if title_hits:
                    score += min(0.65, 0.28 * title_hits)
                if heading_hits:
                    score += min(0.45, 0.18 * heading_hits)
                if body_hits:
                    score += min(0.25, 0.04 * body_hits)

            scores.append(min(score, 1.0))

        return scores


class DocumentPayload(BaseModel):
    id: str
    html: str


DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parent / ".env"
load_env_file(ENV_FILE)
store = DocumentStore()
semantic_engine = SemanticSearchEngine()

def load_seed_documents() -> None:
    if not DATA_DIR.exists():
        try:
            semantic_engine.rebuild(store.list_documents())
        except RuntimeError as exc:
            print(f"Semantic index skipped: {exc}")
        return

    for html_file in sorted(DATA_DIR.glob("*.html")):
        store.upsert(html_file.stem, html_file.read_text(encoding="utf-8"))

    try:
        semantic_engine.rebuild(store.list_documents())
    except RuntimeError as exc:
        print(f"Semantic index skipped: {exc}")


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
    try:
        semantic_engine.rebuild(store.list_documents())
    except RuntimeError as exc:
        print(f"Semantic index skipped after document update: {exc}")
    return {"id": document.id, "title": document.title}


@app.get("/v1/search")
def search_documents(q: str = Query(..., min_length=1)) -> dict:
    return {"query": q, "results": store.search(q)}


@app.get("/v2/search")
def semantic_search_documents(q: str = Query(..., min_length=1)) -> dict:
    try:
        results = semantic_engine.search(q)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": q, "results": results}


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


@app.get("/v2", response_class=HTMLResponse)
def semantic_search_page() -> str:
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>On-Call 语义搜索</title>
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
      margin: 0 0 8px;
      font-size: 28px;
    }
    p {
      margin: 0 0 20px;
      color: #64748b;
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
      background: #0f766e;
      border-color: #0f766e;
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
    <h1>On-Call SOP 语义搜索</h1>
    <p>支持不精确匹配的相似问题检索，例如“服务器挂了”或“黑客攻击”。</p>
    <form id="search-form">
      <input id="query" name="q" type="text" placeholder="输入语义查询，例如 服务器挂了 / 黑客攻击 / 机器学习模型出问题" autocomplete="off">
      <button type="submit">搜索</button>
    </form>
    <div id="status" class="empty">请输入一个问题或描述开始搜索。</div>
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
        status.textContent = "请输入一个问题或描述开始搜索。";
        return;
      }

      status.textContent = "搜索中...";

      const response = await fetch(`/v2/search?q=${encodeURIComponent(query)}`);
      const payload = await response.json();

      if (!response.ok) {
        status.textContent = payload.detail || "语义搜索暂时不可用。";
        return;
      }

      if (!payload.results.length) {
        status.textContent = `没有找到与 "${query}" 语义相关的文档。`;
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
