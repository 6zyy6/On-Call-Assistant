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
from fastapi.responses import HTMLResponse, StreamingResponse
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
}

DOMAIN_HINTS = {
    "backend": ["backend", "service", "api", "gateway", "dependency", "timeout", "circuit breaker"],
    "sre": ["sre", "infrastructure", "kubernetes", "k8s", "ingress", "etcd", "control plane", "gateway", "cloud"],
    "database": ["database", "dba", "mysql", "redis", "postgresql", "replica", "connection pool"],
    "security": ["security", "attack", "injection", "ddos", "vulnerability", "risk", "malicious"],
    "ai": ["ai", "algorithm", "model", "inference", "feature", "gpu", "experiment"],
    "data": ["data", "hadoop", "flink", "kafka", "hdfs", "offline", "realtime"],
}

QUERY_INTENTS = [
    {"triggers": ["server", "service", "down", "outage", "unavailable", "timeout"], "domains": ["backend", "sre"]},
    {"triggers": ["attack", "intrusion", "injection", "ddos", "malicious"], "domains": ["security"]},
    {"triggers": ["model", "ml", "inference", "feature", "gpu", "experiment"], "domains": ["ai"]},
    {"triggers": ["database", "replica", "connection pool", "slow query", "binlog"], "domains": ["database"]},
    {"triggers": ["hadoop", "flink", "kafka", "hdfs", "offline job", "realtime compute"], "domains": ["data"]},
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

    def _base_url(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")

    def _model_name(self) -> str:
        return os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus").strip() or "qwen3.6-plus"

    def _available_files_text(self) -> str:
        documents = store.list_documents()
        if not documents:
            return "No documents are loaded."
        lines = [f"- {document.id}.html: {document.title}" for document in documents]
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return (
            "You are an On-Call assistant.\n"
            "Answer only from SOP documents in data/.\n"
            "Your only tool is readFile(fname), which reads one exact file.\n"
            "Call readFile before relying on document details. Never claim to have read a file unless you actually called the tool.\n"
            "Do not list directories, do not use wildcards, and do not read outside data/.\n"
            "Prefer reading the 1-3 most relevant files, then answer with concise, actionable troubleshooting steps.\n"
            "If the question is vague, choose the closest SOP, explain the assumption briefly, and keep moving.\n"
            "When structure helps, use clean Markdown: headings, numbered steps, bullets, blockquotes, and `inline code`.\n"
            "If the SOP does not support a claim, say that clearly instead of guessing.\n"
            "Available files:\n"
            f"{self._available_files_text()}"
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
        return candidate.read_text(encoding="utf-8")

    def _build_messages(self, history: List[ChatHistoryItem], message: str) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt()}]
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

    def chat(self, message: str, history: List[ChatHistoryItem]) -> dict:
        tools = self._tool_spec()
        messages = self._build_messages(history, message)
        tool_calls_log: List[dict] = []

        with self._lock:
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

        raise RuntimeError("The agent did not finish within the tool-call limit.")

    def stream_chat(self, message: str, history: List[ChatHistoryItem]) -> Iterator[str]:
        tools = self._tool_spec()
        messages = self._build_messages(history, message)

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
  <title>On-Call 鎼滅储</title>
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
    <h1>On-Call SOP 鎼滅储</h1>
    <form id="search-form">
      <input id="query" name="q" type="text" placeholder="杈撳叆鍏抽敭璇嶏紝渚嬪 OOM / 鏁呴殰 / CDN / &" autocomplete="off">
      <button type="submit">鎼滅储</button>
    </form>
    <div id="status" class="empty">璇疯緭鍏ュ叧閿瘝寮€濮嬫悳绱€?/div>
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
        status.textContent = "璇疯緭鍏ュ叧閿瘝寮€濮嬫悳绱€?;
        return;
      }

      status.textContent = "鎼滅储涓?..";
      const response = await fetch(`/v1/search?q=${encodeURIComponent(query)}`);
      const payload = await response.json();

      if (!payload.results.length) {
        status.textContent = `娌℃湁鎵惧埌涓?"${query}" 鐩稿叧鐨勬枃妗ｃ€俙;
        return;
      }

      status.textContent = `鎵惧埌 ${payload.results.length} 鏉＄粨鏋溿€俙;
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
  <title>On-Call 璇箟鎼滅储</title>
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
    <h1>On-Call SOP 璇箟鎼滅储</h1>
    <p>鏀寔涓嶇簿纭尮閰嶇殑鐩镐技闂妫€绱紝渚嬪鈥滄湇鍔″櫒鎸備簡鈥濇垨鈥滈粦瀹㈡敾鍑烩€濄€?/p>
    <form id="search-form">
      <input id="query" name="q" type="text" placeholder="杈撳叆璇箟鏌ヨ锛屼緥濡?鏈嶅姟鍣ㄦ寕浜?/ 榛戝鏀诲嚮 / 鏈哄櫒瀛︿範妯″瀷鍑洪棶棰? autocomplete="off">
      <button type="submit">鎼滅储</button>
    </form>
    <div id="status" class="empty">璇疯緭鍏ヤ竴涓棶棰樻垨鎻忚堪寮€濮嬫悳绱€?/div>
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
        status.textContent = "璇疯緭鍏ヤ竴涓棶棰樻垨鎻忚堪寮€濮嬫悳绱€?;
        return;
      }

      status.textContent = "鎼滅储涓?..";

      const response = await fetch(`/v2/search?q=${encodeURIComponent(query)}`);
      const payload = await response.json();

      if (!response.ok) {
        status.textContent = payload.detail || "璇箟鎼滅储鏆傛椂涓嶅彲鐢ㄣ€?;
        return;
      }

      if (!payload.results.length) {
        status.textContent = `娌℃湁鎵惧埌涓?"${query}" 璇箟鐩稿叧鐨勬枃妗ｃ€俙;
        return;
      }

      status.textContent = `鎵惧埌 ${payload.results.length} 鏉＄粨鏋溿€俙;
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
    .message.tool {
      border-color: #dbe4f0;
      background: #f8fafc;
      color: #475569;
      font-size: 14px;
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

      const assistantMessage = addMessage("assistant", "正在思考...", true);
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
            addMessage("tool", eventPayload.message || "", false, assistantMessage.item);
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

