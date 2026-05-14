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
from typing import Dict, Iterator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
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
    "server down": ["service unavailable", "service outage", "service crash", "service timeout", "backend incident", "infrastructure incident", "kubernetes", "ingress"],
    "outage": ["unavailable", "down", "interruption", "timeout"],
    "attack": ["security incident", "intrusion", "sql injection", "ddos", "malicious traffic", "waf"],
    "model issue": ["model serving incident", "inference failure", "feature service incident", "gpu incident", "quality drop", "recommendation service"],
    "服务器挂了": ["服务不可用", "服务超时", "服务宕机", "后端故障", "基础设施故障", "kubernetes", "ingress"],
    "挂了": ["不可用", "宕机", "中断", "超时"],
    "黑客攻击": ["安全事件", "入侵", "ddos", "sql注入", "漏洞利用", "恶意流量"],
    "攻击": ["入侵", "恶意流量", "ddos", "注入"],
    "机器学习模型出问题": ["模型推理异常", "模型服务故障", "推荐质量下降", "gpu故障", "特征服务异常"],
    "模型出问题": ["模型服务故障", "推理异常", "推荐质量下降", "gpu故障"],
    "oom": ["outofmemoryerror", "memory leak", "pod restart", "内存泄漏", "内存溢出"],
}

DOMAIN_HINTS = {
    "backend": ["backend", "service", "api", "gateway", "dependency", "timeout", "circuit breaker", "后端", "服务", "接口", "超时", "降级", "熔断", "pod", "oom", "内存"],
    "sre": ["sre", "infrastructure", "kubernetes", "k8s", "ingress", "etcd", "control plane", "gateway", "cloud", "基础设施", "集群", "监控", "告警", "容量", "节点"],
    "database": ["database", "dba", "mysql", "redis", "postgresql", "replica", "connection pool", "数据库", "主从", "延迟", "慢查询", "连接池", "binlog"],
    "security": ["security", "attack", "injection", "ddos", "vulnerability", "risk", "malicious", "安全", "攻击", "入侵", "漏洞", "恶意", "waf"],
    "ai": ["ai", "algorithm", "model", "inference", "feature", "gpu", "experiment", "算法", "模型", "推理", "特征", "推荐", "质量下降", "gpu"],
    "data": ["data", "hadoop", "flink", "kafka", "hdfs", "offline", "realtime", "数据", "离线", "实时", "任务", "spark", "etl"],
    "frontend": ["frontend", "web", "cdn", "页面", "白屏", "资源加载", "兼容性", "前端"],
    "mobile": ["mobile", "app", "crash", "push", "热修复", "移动端", "崩溃率", "推送"],
}

QUERY_INTENTS = [
    {"triggers": ["server", "service", "down", "outage", "unavailable", "timeout", "服务器", "服务", "挂了", "宕机", "不可用"], "domains": ["backend", "sre"]},
    {"triggers": ["attack", "intrusion", "injection", "ddos", "malicious", "黑客", "攻击", "入侵", "漏洞", "恶意"], "domains": ["security"]},
    {"triggers": ["model", "ml", "inference", "feature", "gpu", "experiment", "机器学习", "模型", "推理", "特征", "推荐"], "domains": ["ai"]},
    {"triggers": ["database", "replica", "connection pool", "slow query", "binlog", "数据库", "主从", "延迟", "慢查询", "连接池"], "domains": ["database"]},
    {"triggers": ["hadoop", "flink", "kafka", "hdfs", "offline job", "realtime compute", "数据", "etl", "spark"], "domains": ["data"]},
    {"triggers": ["oom", "outofmemoryerror", "内存泄漏", "内存溢出"], "domains": ["backend", "sre"]},
    {"triggers": ["cdn", "白屏", "资源加载"], "domains": ["frontend"]},
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
    normalized = normalize_text(query).lower()
    if not normalized:
        return []

    expansions: List[str] = []
    seen = {normalized}

    for phrase, related in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            for item in related:
                candidate = normalize_text(item).lower()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    expansions.append(candidate)

    if any(token in normalized for token in ("server", "service")):
        for candidate in ("service incident", "service unavailable", "backend service", "infrastructure"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    if any(token in normalized for token in ("down", "outage")):
        for candidate in ("incident", "unavailable", "interruption", "timeout"):
            if candidate not in seen:
                seen.add(candidate)
                expansions.append(candidate)

    return expansions


def build_hybrid_query(query: str) -> str:
    parts = [normalize_text(query)] + expand_query(query)
    return " ".join(part for part in parts if part)


def query_contains_trigger(combined_query: str, combined_tokens: List[str], trigger: str) -> bool:
    normalized_trigger = normalize_text(trigger).lower()
    if not normalized_trigger:
        return False
    if " " in normalized_trigger:
        return normalized_trigger in combined_query
    return normalized_trigger in combined_tokens


@dataclass
class StoredDocument:
    id: str
    title: str
    html: str
    text: str
    heading_text: str
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


def extract_heading_text(html: str, title: str) -> str:
    cleaned_html = IGNORED_BLOCK_PATTERN.sub(" ", html)
    headings = [title]
    for match in re.finditer(r"<(h[1-3])[^>]*>(.*?)</\1>", cleaned_html, re.IGNORECASE | re.DOTALL):
        content = strip_tags(match.group(2))
        if content:
            headings.append(content)
    return normalize_text(" ".join(headings))


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
            heading_text=extract_heading_text(html, title or doc_id),
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

        normalized_query = normalize_text(query).lower()

        with self._lock:
            documents = list(self._documents.values())
            doc_count = len(documents)
            if doc_count == 0:
                return []

            average_length = self._total_terms / doc_count if self._total_terms else 0.0
            results = []
            for document in documents:
                if not self._contains_keyword(document, normalized_query):
                    continue
                score = self._bm25_score(document, query_tokens, doc_count, average_length)
                if score <= 0:
                    score = 0.01
                score += self._document_metadata_boost(document, normalized_query, query_tokens)
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

    def _contains_keyword(self, document: StoredDocument, normalized_query: str) -> bool:
        if not normalized_query:
            return False
        haystacks = [document.title.lower(), document.heading_text.lower(), document.text.lower()]
        return any(normalized_query in haystack for haystack in haystacks)

    def _document_metadata_boost(self, document: StoredDocument, normalized_query: str, query_tokens: List[str]) -> float:
        heading_text = document.heading_text.lower()
        title_text = document.title.lower()
        body_text = document.text.lower()
        score = 0.0

        if normalized_query and normalized_query in title_text:
            score += 0.9
        if normalized_query and normalized_query in heading_text:
            score += 1.2
        if normalized_query and normalized_query in body_text:
            score += 0.35
            score += 0.08 * body_text.count(normalized_query)

        for token in query_tokens:
            if token and token in heading_text:
                score += 0.1
            elif token and token in title_text:
                score += 0.06

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
            if not self._chunks:
                return []
            if self._index is None:
                return self._offline_search(query, limit)

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

    def _offline_search(self, query: str, limit: int = 10) -> List[dict]:
        expanded_terms = expand_query(query)
        hybrid_query = build_hybrid_query(query)
        bm25_scores = self._score_chunks_bm25(hybrid_query)
        keyword_scores = self._score_heading_matches(query, expanded_terms)
        intent_scores = self._score_intent_boosts(query, expanded_terms)
        phrase_scores = self._score_phrase_matches(query, expanded_terms)

        candidate_indices = {
            index
            for index, score in sorted(
                enumerate(bm25_scores),
                key=lambda item: item[1],
                reverse=True,
            )[: min(max(limit * 8, 16), len(self._chunks))]
            if score > 0
        }
        candidate_indices.update(index for index, score in enumerate(keyword_scores) if score > 0)
        candidate_indices.update(index for index, score in enumerate(intent_scores) if score > 0)
        candidate_indices.update(index for index, score in enumerate(phrase_scores) if score > 0)

        if not candidate_indices:
            candidate_indices = set(range(min(len(self._chunks), max(limit * 4, 12))))

        max_bm25 = max((bm25_scores[index] for index in candidate_indices), default=1.0)
        doc_results: Dict[str, dict] = {}
        for chunk_index in candidate_indices:
            chunk = self._chunks[chunk_index]
            normalized_bm25 = bm25_scores[chunk_index] / max_bm25 if max_bm25 > 0 else 0.0
            final_score = (
                0.45 * normalized_bm25
                + 0.20 * keyword_scores[chunk_index]
                + 0.20 * intent_scores[chunk_index]
                + 0.15 * phrase_scores[chunk_index]
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
        combined_tokens = tokenize(combined_query)
        target_domains: List[str] = []

        for intent in QUERY_INTENTS:
            if any(query_contains_trigger(combined_query, combined_tokens, trigger) for trigger in intent["triggers"]):
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

    def _score_phrase_matches(self, query: str, expanded_terms: List[str]) -> List[float]:
        phrases = [normalize_text(query).lower()] + [term.lower() for term in expanded_terms]
        scores: List[float] = []
        for chunk in self._chunks:
            haystack = chunk.context_text.lower()
            score = 0.0
            for phrase in phrases:
                if phrase and phrase in haystack:
                    score += 0.4
            scores.append(min(score, 1.0))
        return scores


class DocumentPayload(BaseModel):
    id: str
    html: str


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem] = Field(default_factory=list)


def encode_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class OnCallChatAgent:
    def __init__(self) -> None:
        self._lock = RLock()
        self._retrieval_cache: Dict[str, List[dict]] = {}
        self._file_cache: Dict[str, tuple[int, str]] = {}

    def _tool_spec(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "readFile",
                    "description": "Read one exact file from the data directory, such as sop-001.html.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fname": {
                                "type": "string",
                                "description": "Exact file name under data/, for example sop-001.html",
                            }
                        },
                        "required": ["fname"],
                    },
                },
            }
        ]

    def _api_key(self) -> str:
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY. Set it before using /v3 chat.")
        return api_key

    def _has_api_key(self) -> bool:
        return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())

    def _base_url(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

    def _model_name(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus").strip() or "qwen3.6-plus"

    def _cache_retrieval(self, key: str, candidates: List[dict]) -> List[dict]:
        self._retrieval_cache[key] = [dict(item) for item in candidates]
        if len(self._retrieval_cache) > 128:
            oldest_key = next(iter(self._retrieval_cache))
            del self._retrieval_cache[oldest_key]
        return [dict(item) for item in candidates]

    def _rag_candidates(self, message: str, history: List[ChatHistoryItem], limit: int = 4) -> List[dict]:
        query_parts: List[str] = []
        for item in history[-4:]:
            if item.role.strip().lower() == "user":
                query_parts.append(item.content)
        query_parts.append(message)
        query = normalize_text(" ".join(part for part in query_parts if part))
        if not query:
            return []

        cache_key = f"{limit}:{query.lower()}"
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        merged: Dict[str, dict] = {}
        keyword_results = store.search(query, limit=limit)
        for result in keyword_results:
            merged[result["id"]] = {
                "id": result["id"],
                "title": result["title"],
                "snippet": result.get("snippet", ""),
                "score": float(result.get("score", 0.0)),
                "source": "keyword",
            }

        should_run_semantic = True
        if keyword_results:
            top_keyword_score = float(keyword_results[0].get("score", 0.0))
            if top_keyword_score >= 2.0 and len(keyword_results) >= min(3, limit):
                should_run_semantic = False

        if should_run_semantic:
            try:
                for result in semantic_engine.search(query, limit=limit):
                    existing = merged.get(result["id"])
                    score = float(result.get("score", 0.0))
                    if existing is None or score > existing["score"]:
                        merged[result["id"]] = {
                            "id": result["id"],
                            "title": result["title"],
                            "snippet": result.get("snippet", ""),
                            "score": score,
                            "source": "semantic",
                        }
            except RuntimeError:
                pass

        candidates = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        if candidates:
            return self._cache_retrieval(cache_key, candidates[:limit])

        fallback_candidates = [
            {
                "id": document.id,
                "title": document.title,
                "snippet": "",
                "score": 0.0,
                "source": "fallback",
            }
            for document in store.list_documents()[: min(limit, 4)]
        ]
        return self._cache_retrieval(cache_key, fallback_candidates)

    def _candidate_files_text(self, candidates: List[dict]) -> str:
        if not candidates:
            return "No candidate files are available."

        lines = []
        for index, candidate in enumerate(candidates, start=1):
            snippet = normalize_text(candidate.get("snippet", ""))
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            lines.append(f"{index}. {candidate['id']}.html - {candidate['title']}")
            if snippet:
                lines.append(f"   snippet: {snippet}")
        return "\n".join(lines)

    def _system_prompt(self, candidates: List[dict]) -> str:
        return (
            "You are an On-Call assistant.\n"
            "Answer only from SOP documents in data/.\n"
            "Your only tool is readFile(fname), which reads one exact file.\n"
            "Call readFile before relying on document details. Never claim to have read a file unless you actually called the tool.\n"
            "Do not list directories, do not use wildcards, and do not read outside data/.\n"
            "The system may provide retrieved candidate files. Prefer those candidates first when they match the user request.\n"
            "Prefer reading the 1-3 most relevant files, then answer with concise, actionable troubleshooting steps.\n"
            "If the question is vague, choose the closest SOP, explain the assumption briefly, and keep moving.\n"
            "When structure helps, use clean Markdown: headings, numbered steps, bullets, blockquotes, and `inline code`.\n"
            "If the SOP does not support a claim, say that clearly instead of guessing.\n"
            "Retrieved candidate files:\n"
            f"{self._candidate_files_text(candidates)}"
        )

    def _chat_completion(self, messages: List[dict], tools: List[dict]) -> dict:
        payload = {
            "model": self._model_name(),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        request = Request(
            f"{self._base_url()}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DashScope chat request failed: HTTP {exc.code}. {details}".strip()) from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope chat request failed: {exc.reason}") from exc

    def _stream_chat_completion(self, messages: List[dict], tools: List[dict]) -> Iterator[dict]:
        payload = {
            "model": self._model_name(),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "stream": True,
        }
        request = Request(
            f"{self._base_url()}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    yield json.loads(data)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DashScope chat request failed: HTTP {exc.code}. {details}".strip()) from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope chat request failed: {exc.reason}") from exc

    def _read_file(self, fname: str) -> str:
        normalized = fname.strip().replace("\\", "/")
        if not normalized:
            raise RuntimeError("fname is required")
        if "*" in normalized or "?" in normalized:
            raise RuntimeError("Wildcards are not allowed")
        candidate = (DATA_DIR / normalized).resolve()
        data_root = DATA_DIR.resolve()
        if data_root not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"File not found or access denied: {fname}")
        stat = candidate.stat()
        cache_key = str(candidate)
        cached = self._file_cache.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns:
            return cached[1]
        content = candidate.read_text(encoding="utf-8")
        self._file_cache[cache_key] = (stat.st_mtime_ns, content)
        if len(self._file_cache) > 128:
            oldest_key = next(iter(self._file_cache))
            del self._file_cache[oldest_key]
        return content

    def _build_messages(self, history: List[ChatHistoryItem], message: str, candidates: List[dict]) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt(candidates)}]
        for item in history:
            role = item.role.strip().lower()
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": item.content})
        messages.append({"role": "user", "content": message})
        return messages

    def _execute_tool_call(self, tool_call: dict) -> tuple[str, List[dict], dict]:
        function_payload = tool_call.get("function", {})
        name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "{}")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"fname": ""}

        fname = str(arguments.get("fname", "")).strip()
        events = [
            {
                "type": "tool_call",
                "name": name,
                "fname": fname,
                "status": "started",
                "message": f"正在读取 {fname}" if fname else "正在读取文件",
            }
        ]

        if name != "readFile":
            tool_output = f"Unsupported tool: {name}"
            events.append(
                {
                    "type": "tool_call",
                    "name": name,
                    "fname": fname,
                    "status": "failed",
                    "message": f"读取 {fname or '文件'} 失败：不支持的工具",
                }
            )
        else:
            try:
                tool_output = self._read_file(fname)
                events.append(
                    {
                        "type": "tool_call",
                        "name": name,
                        "fname": fname,
                        "status": "completed",
                        "message": f"已读取 {fname}",
                    }
                )
            except RuntimeError as exc:
                tool_output = str(exc)
                events.append(
                    {
                        "type": "tool_call",
                        "name": name,
                        "fname": fname,
                        "status": "failed",
                        "message": f"读取 {fname or '文件'} 失败：{exc}",
                    }
                )

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "content": tool_output,
        }
        return tool_output, events, tool_message

    def _read_documents_for_offline_answer(self, message: str, history: List[ChatHistoryItem]) -> tuple[List[dict], List[dict]]:
        candidates = self._rag_candidates(message, history)
        normalized_message = normalize_text(message).lower()
        read_limit = 3 if any(term in normalized_message for term in ["p0", "流程", "升级", "响应", "综合"]) else 1
        selected = candidates[:read_limit] or [
            {"id": document.id, "title": document.title, "snippet": "", "score": 0.0, "source": "fallback"}
            for document in store.list_documents()[:1]
        ]

        tool_logs: List[dict] = []
        loaded_documents: List[dict] = []
        for index, candidate in enumerate(selected):
            tool_call = {
                "id": f"offline-read-{index}",
                "type": "function",
                "function": {
                    "name": "readFile",
                    "arguments": json.dumps({"fname": f"{candidate['id']}.html"}, ensure_ascii=False),
                },
            }
            content, events, _ = self._execute_tool_call(tool_call)
            tool_logs.extend(events)
            loaded_documents.append(
                {
                    "id": candidate["id"],
                    "title": candidate["title"],
                    "html": content,
                }
            )
        return loaded_documents, tool_logs

    def _rank_chunk_texts(self, message: str, document_id: str, title: str, html: str, limit: int = 3) -> List[str]:
        chunks = extract_document_chunks(html, title, document_id)
        if not chunks:
            return []

        query = build_hybrid_query(message)
        query_tokens = tokenize(query)
        normalized_query = normalize_text(message).lower()
        expanded_terms = [term.lower() for term in expand_query(message)]
        ranked: List[tuple[float, str]] = []
        for chunk in chunks:
            heading_text = normalize_text(f"{chunk.doc_title} {chunk.heading}").lower()
            body_text = chunk.context_text.lower()
            token_overlap = sum(chunk.term_freq.get(token, 0) for token in query_tokens)
            phrase_hits = sum(1 for phrase in [normalized_query] + expanded_terms if phrase and phrase in body_text)
            heading_hits = sum(1 for token in query_tokens if token and token in heading_text)
            score = (1.4 * token_overlap) + (1.2 * phrase_hits) + (0.8 * heading_hits)
            ranked.append((score, chunk.text))

        ranked.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        results: List[str] = []
        for _, text in ranked:
            summary = self._shorten_guidance(text)
            if not summary or summary in seen:
                continue
            seen.add(summary)
            results.append(summary)
            if len(results) >= limit:
                break
        return results

    def _shorten_guidance(self, text: str, limit: int = 150) -> str:
        normalized = normalize_text(text)
        if len(normalized) <= limit:
            return normalized

        parts = [part.strip() for part in re.split(r"[。！？；]", normalized) if part.strip()]
        snippet = ""
        for part in parts:
            candidate = f"{snippet}；{part}" if snippet else part
            if len(candidate) > limit:
                break
            snippet = candidate
        return snippet or (normalized[: limit - 3] + "...")

    def _build_offline_reply(self, message: str, documents: List[dict]) -> str:
        if not documents:
            return "我暂时没有找到可参考的 SOP 文档。"

        normalized_message = normalize_text(message).lower()
        sections: List[str] = []
        if len(documents) == 1:
            document = documents[0]
            sections.append(f"## 处理建议\n\n已参考 `{document['id']}.html`（{document['title']}）。")
            guidance = self._rank_chunk_texts(message, document["id"], document["title"], document["html"])
            if guidance:
                sections.extend(f"{index}. {item}" for index, item in enumerate(guidance, start=1))
            else:
                sections.append("1. 先核对告警范围、影响面和最近变更。")
                sections.append("2. 再根据 SOP 中对应场景执行排查与止损。")
        else:
            sections.append("## 综合处理建议\n")
            for document in documents:
                guidance = self._rank_chunk_texts(message, document["id"], document["title"], document["html"], limit=2)
                sections.append(f"### `{document['id']}.html` - {document['title']}")
                if guidance:
                    sections.extend(f"- {item}" for item in guidance)

        if any(term in normalized_message for term in ["p0", "升级", "响应流程", "响应"]):
            sections.append("\n## 升级提醒\n")
            sections.append("- 先确认影响范围、持续时间和是否命中核心链路。")
            sections.append("- 若已达到 P0/P1 级别，立即按值班链路升级，并同步相关负责人。")

        return "\n".join(sections).strip()

    def _offline_chat(self, message: str, history: List[ChatHistoryItem]) -> dict:
        documents, tool_logs = self._read_documents_for_offline_answer(message, history)
        return {
            "reply": self._build_offline_reply(message, documents),
            "tool_calls": tool_logs,
        }

    def _offline_stream_chat(self, message: str, history: List[ChatHistoryItem]) -> Iterator[str]:
        documents, tool_logs = self._read_documents_for_offline_answer(message, history)
        for event in tool_logs:
            yield encode_sse_event(event)
        yield encode_sse_event({"type": "status", "message": "助手正在整理 SOP 结论..."})
        reply = self._build_offline_reply(message, documents)
        yield encode_sse_event({"type": "reply_delta", "delta": reply})
        yield encode_sse_event({"type": "done", "reply": reply})

    def chat(self, message: str, history: List[ChatHistoryItem]) -> dict:
        if not self._has_api_key():
            return self._offline_chat(message, history)
        tools = self._tool_spec()
        candidates = self._rag_candidates(message, history)
        messages = self._build_messages(history, message, candidates)
        tool_calls_log: List[dict] = []

        with self._lock:
            try:
                for _ in range(4):
                    response_payload = self._chat_completion(messages, tools)
                    choices = response_payload.get("choices") or []
                    if not choices:
                        raise RuntimeError("DashScope chat response did not include any choices.")
                    choice = choices[0]
                    assistant_message = choice.get("message", {})
                    tool_calls = assistant_message.get("tool_calls") or []
                    content = assistant_message.get("content") or ""

                    if tool_calls:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls,
                            }
                        )
                        for tool_call in tool_calls:
                            _, events, tool_message = self._execute_tool_call(tool_call)
                            tool_calls_log.extend(events)
                            messages.append(tool_message)
                        continue

                    return {
                        "reply": content.strip() or "我暂时还没有整理出可靠结论。",
                        "tool_calls": tool_calls_log,
                    }
            except RuntimeError:
                return self._offline_chat(message, history)

        raise RuntimeError("The agent did not finish within the tool-call limit.")

    def stream_chat(self, message: str, history: List[ChatHistoryItem]) -> Iterator[str]:
        if not self._has_api_key():
            yield from self._offline_stream_chat(message, history)
            return
        tools = self._tool_spec()
        candidates = self._rag_candidates(message, history)
        messages = self._build_messages(history, message, candidates)

        try:
            with self._lock:
                yield encode_sse_event({"type": "status", "message": "助手正在思考..."})
                for _ in range(4):
                    content_parts: List[str] = []
                    pending_tool_calls: Dict[int, dict] = {}

                    for chunk in self._stream_chat_completion(messages, tools):
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        delta_content = delta.get("content")
                        if delta_content:
                            content_parts.append(delta_content)
                            yield encode_sse_event({"type": "reply_delta", "delta": delta_content})

                        for tool_call in delta.get("tool_calls") or []:
                            index = int(tool_call.get("index", 0))
                            entry = pending_tool_calls.setdefault(
                                index,
                                {
                                    "id": tool_call.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tool_call.get("id"):
                                entry["id"] = tool_call["id"]
                            function_payload = tool_call.get("function", {})
                            if function_payload.get("name"):
                                entry["function"]["name"] = function_payload["name"]
                            if function_payload.get("arguments"):
                                entry["function"]["arguments"] += function_payload["arguments"]

                    if pending_tool_calls:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "".join(content_parts),
                                "tool_calls": [pending_tool_calls[index] for index in sorted(pending_tool_calls)],
                            }
                        )
                        for index in sorted(pending_tool_calls):
                            _, events, tool_message = self._execute_tool_call(pending_tool_calls[index])
                            for event in events:
                                yield encode_sse_event(event)
                            messages.append(tool_message)
                        continue

                    reply = "".join(content_parts).strip()
                    yield encode_sse_event({"type": "done", "reply": reply})
                    return

            yield encode_sse_event({"type": "error", "message": "助手在工具调用轮次限制内未完成回答。"})
        except RuntimeError as exc:
            if self._has_api_key():
                yield from self._offline_stream_chat(message, history)
            else:
                yield encode_sse_event({"type": "error", "message": str(exc)})


DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parent / ".env"
load_env_file(ENV_FILE)
store = DocumentStore()
semantic_engine = SemanticSearchEngine()
chat_agent = OnCallChatAgent()

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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = (DATA_DIR / f"{payload.id}.html").resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in target.parents:
        raise HTTPException(status_code=400, detail="Document id is invalid")
    target.write_text(payload.html, encoding="utf-8")
    document = store.upsert(payload.id, payload.html)
    try:
        semantic_engine.rebuild(store.list_documents())
    except RuntimeError as exc:
        print(f"Semantic index skipped after document update: {exc}")
    return JSONResponse(status_code=201, content={"id": document.id, "title": document.title})


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


@app.post("/v3/chat")
def chat_with_agent(payload: ChatRequest) -> dict:
    try:
        return chat_agent.chat(payload.message, payload.history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v3/chat/stream")
def stream_chat_with_agent(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_agent.stream_chat(payload.message, payload.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1", response_class=HTMLResponse)
def search_page() -> str:
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>On-Call 关键词搜索</title>
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
    <h1>On-Call SOP 关键词搜索</h1>
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
    <p>支持不要求原词精确出现的相关问题检索，例如“服务器挂了”“黑客攻击”“机器学习模型出问题”。</p>
    <form id="search-form">
      <input id="query" name="q" type="text" placeholder="输入语义查询，例如：服务器挂了 / 黑客攻击 / 机器学习模型出问题" autocomplete="off">
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


@app.get("/v3", response_class=HTMLResponse)
def chat_page() -> str:
    return r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>On-Call 助手</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #f3f6fb;
      color: #0f172a;
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 24px 20px 28px;
      display: grid;
      gap: 16px;
    }
    .hero {
      display: grid;
      gap: 6px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }
    .subtitle {
      margin: 0;
      color: #64748b;
    }
    .chat-shell {
      display: grid;
      gap: 14px;
      min-height: 72vh;
      background: #ffffff;
      border: 1px solid #dbe4f0;
      border-radius: 8px;
      padding: 16px;
    }
    #history {
      min-height: 480px;
      display: grid;
      align-content: start;
      gap: 12px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .empty {
      color: #64748b;
      padding: 8px 0;
    }
    .message {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 14px;
      background: #fff;
    }
    .message.user {
      border-color: #bfdbfe;
      background: #eff6ff;
    }
    .message.assistant {
      border-color: #bbf7d0;
      background: #f0fdf4;
    }
    .message.assistant.streaming {
      box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.18);
    }
    .label {
      margin-bottom: 6px;
      font-size: 12px;
      color: #64748b;
    }
    .content {
      line-height: 1.72;
      word-break: break-word;
    }
    .content p {
      margin: 0 0 10px;
    }
    .content p:last-child {
      margin-bottom: 0;
    }
    .content h1, .content h2, .content h3 {
      margin: 0 0 10px;
      line-height: 1.4;
    }
    .content h1 { font-size: 22px; }
    .content h2 { font-size: 18px; }
    .content h3 { font-size: 16px; }
    .content ul, .content ol {
      margin: 0 0 10px 20px;
      padding: 0;
    }
    .content li {
      margin: 0 0 6px;
    }
    .content blockquote {
      margin: 0 0 10px;
      padding: 8px 12px;
      border-left: 3px solid #94a3b8;
      background: rgba(148, 163, 184, 0.08);
      color: #334155;
    }
    .content hr {
      border: 0;
      border-top: 1px solid #cbd5e1;
      margin: 12px 0;
    }
    .content pre {
      margin: 10px 0;
      padding: 12px;
      border-radius: 6px;
      background: #0f172a;
      color: #e2e8f0;
      overflow-x: auto;
    }
    .content code {
      font-family: Consolas, monospace;
      background: rgba(15, 23, 42, 0.08);
      border-radius: 4px;
      padding: 1px 4px;
    }
    .content pre code {
      background: transparent;
      padding: 0;
    }
    .content a {
      color: #2563eb;
    }
    .tool-trace {
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
      padding: 10px 12px;
      border-radius: 6px;
      background: rgba(15, 23, 42, 0.05);
      color: #475569;
      font-size: 13px;
    }
    .tool-trace:empty {
      display: none;
    }
    .tool-event {
      line-height: 1.5;
    }
    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }
    .composer > div {
      min-width: 0;
    }
    textarea, button {
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 15px;
    }
    textarea {
      width: 100%;
      min-height: 104px;
      max-height: 220px;
      padding: 12px 14px;
      resize: vertical;
      font-family: inherit;
      line-height: 1.5;
    }
    button {
      min-width: 120px;
      height: 46px;
      padding: 0 18px;
      background: #2563eb;
      border-color: #2563eb;
      color: white;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.7;
      cursor: wait;
    }
    .hint {
      font-size: 13px;
      color: #64748b;
    }
    @media (max-width: 720px) {
      .composer {
        grid-template-columns: 1fr;
      }
      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>On-Call 助手</h1>
      <p class="subtitle">通过自然语言提问，助手会按需读取 `data/` 下的 SOP，并实时展示工具调用过程与回复。</p>
    </div>
    <section class="chat-shell">
      <div id="history">
        <div class="empty">还没有对话。比如问：`Ingress 大量 502 应该先查什么？`</div>
      </div>
      <form id="chat-form" class="composer">
        <div>
          <textarea id="message" placeholder="输入 On-Call 问题，例如：支付接口大量超时，我先排查什么？"></textarea>
          <div class="hint">Enter 发送，Shift+Enter 换行</div>
        </div>
        <button id="send" type="submit">发送</button>
      </form>
    </section>
  </main>
  <script>
    const historyEl = document.getElementById("history");
    const form = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");
    const sendButton = document.getElementById("send");
    const conversation = [];

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function removeEmptyState() {
      const empty = historyEl.querySelector(".empty");
      if (empty) empty.remove();
    }

    function renderEmptyState() {
      if (historyEl.children.length === 0) {
        historyEl.innerHTML = '<div class="empty">还没有对话。比如问：`Ingress 大量 502 应该先查什么？`</div>';
      }
    }

    function normalizeMarkdownSource(source) {
      return source.replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\n{3,}/g, "\n\n");
    }

    function renderInlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/~~([^~]+)~~/g, "<del>$1</del>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>");
    }

    function renderMarkdown(source) {
      const normalized = normalizeMarkdownSource(source);
      const lines = normalized.split("\n");
      const blocks = [];
      let inCode = false;
      let codeLines = [];
      let listType = "";
      let listItems = [];
      let quoteLines = [];

      function flushList() {
        if (!listItems.length) return;
        const tag = listType === "ol" ? "ol" : "ul";
        blocks.push(`<${tag}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
        listType = "";
        listItems = [];
      }

      function flushQuote() {
        if (!quoteLines.length) return;
        blocks.push(`<blockquote>${quoteLines.map((item) => `<p>${renderInlineMarkdown(item)}</p>`).join("")}</blockquote>`);
        quoteLines = [];
      }

      function flushCode() {
        if (!inCode) return;
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
        codeLines = [];
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (line.startsWith("```")) {
          flushQuote();
          flushList();
          if (inCode) {
            flushCode();
          } else {
            inCode = true;
            codeLines = [];
          }
          continue;
        }
        if (inCode) {
          codeLines.push(rawLine);
          continue;
        }

        const trimmed = line.trim();
        if (!trimmed) {
          flushQuote();
          flushList();
          continue;
        }

        const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
        const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
        const quoteMatch = trimmed.match(/^>\s?(.*)$/);

        if (quoteMatch) {
          flushList();
          quoteLines.push(quoteMatch[1]);
          continue;
        }

        flushQuote();
        if (orderedMatch) {
          if (listType && listType !== "ol") flushList();
          listType = "ol";
          listItems.push(orderedMatch[1]);
          continue;
        }
        if (bulletMatch) {
          if (listType && listType !== "ul") flushList();
          listType = "ul";
          listItems.push(bulletMatch[1]);
          continue;
        }

        flushList();
        const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
        if (headingMatch) {
          const level = headingMatch[1].length;
          blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
          continue;
        }
        if (/^(-{3,}|\*{3,})$/.test(trimmed)) {
          blocks.push("<hr>");
          continue;
        }

        blocks.push(`<p>${renderInlineMarkdown(trimmed)}</p>`);
      }

      flushQuote();
      flushList();
      flushCode();
      return blocks.join("");
    }

    function addMessage(role, content, useMarkdown = false, beforeNode = null) {
      removeEmptyState();
      const item = document.createElement("div");
      item.className = `message ${role}`;

      const label = document.createElement("div");
      label.className = "label";
      label.textContent = role === "user" ? "用户" : role === "assistant" ? "助手" : "工具";

      const body = document.createElement("div");
      body.className = "content";
      if (useMarkdown) {
        body.innerHTML = renderMarkdown(content);
      } else {
        body.textContent = content;
      }

      item.append(label, body);
      if (beforeNode && beforeNode.parentNode === historyEl) {
        historyEl.insertBefore(item, beforeNode);
      } else {
        historyEl.appendChild(item);
      }
      historyEl.scrollTop = historyEl.scrollHeight;
      return { item, body };
    }

    function addAssistantMessage() {
      removeEmptyState();
      const item = document.createElement("div");
      item.className = "message assistant";

      const label = document.createElement("div");
      label.className = "label";
      label.textContent = "助手";

      const trace = document.createElement("div");
      trace.className = "tool-trace";

      const body = document.createElement("div");
      body.className = "content";
      body.innerHTML = renderMarkdown("正在思考...");

      item.append(label, trace, body);
      historyEl.appendChild(item);
      historyEl.scrollTop = historyEl.scrollHeight;
      return { item, body, trace };
    }

    function addToolEvent(assistantMessage, content) {
      const event = document.createElement("div");
      event.className = "tool-event";
      event.textContent = content;
      assistantMessage.trace.appendChild(event);
      historyEl.scrollTop = historyEl.scrollHeight;
    }

    async function consumeEventStream(response, onEvent) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          const dataLines = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim());
          for (const line of dataLines) {
            if (line) onEvent(JSON.parse(line));
          }
        }
      }
    }

    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message) return;

      addMessage("user", message);
      conversation.push({ role: "user", content: message });
      messageInput.value = "";
      sendButton.disabled = true;

      const assistantMessage = addAssistantMessage();
      assistantMessage.item.classList.add("streaming");

      let reply = "";
      let paintQueued = false;

      function paintReply(isFinal = false) {
        paintQueued = false;
        assistantMessage.body.innerHTML = renderMarkdown(reply || (isFinal ? "我暂时还没有整理出可靠结论。" : ""));
        if (isFinal) assistantMessage.item.classList.remove("streaming");
        historyEl.scrollTop = historyEl.scrollHeight;
      }

      function queuePaint() {
        if (paintQueued) return;
        paintQueued = true;
        window.requestAnimationFrame(() => paintReply(false));
      }

      try {
        const response = await fetch("/v3/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, history: conversation.slice(0, -1) }),
        });

        if (!response.ok || !response.body) {
          assistantMessage.body.textContent = "请求失败。";
          assistantMessage.item.classList.remove("streaming");
          conversation.push({ role: "assistant", content: "请求失败。" });
          return;
        }

        assistantMessage.body.innerHTML = "";
        await consumeEventStream(response, (eventPayload) => {
          if (eventPayload.type === "reply_delta") {
            reply += eventPayload.delta || "";
            queuePaint();
            return;
          }

          if (eventPayload.type === "tool_call" || eventPayload.type === "status") {
            addToolEvent(assistantMessage, eventPayload.message || "");
            return;
          }

          if (eventPayload.type === "error") {
            reply = eventPayload.message || "请求失败。";
            assistantMessage.body.innerHTML = renderMarkdown(reply);
            assistantMessage.item.classList.remove("streaming");
            return;
          }

          if (eventPayload.type === "done") {
            reply = eventPayload.reply || reply;
            paintReply(true);
          }
        });

        conversation.push({ role: "assistant", content: reply || "我暂时还没有整理出可靠结论。" });
      } catch (error) {
        assistantMessage.body.textContent = "请求失败，请检查服务或网络配置。";
        assistantMessage.item.classList.remove("streaming");
        conversation.push({ role: "assistant", content: "请求失败，请检查服务或网络配置。" });
      } finally {
        sendButton.disabled = false;
        messageInput.focus();
        renderEmptyState();
      }
    });
  </script>
</body>
</html>
"""

@app.get("/")
def root() -> dict:
    return {"message": "Visit /v1 for keyword search, /v2 for semantic search, and /v3 for chat."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

