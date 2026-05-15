import json
import math
import os
from collections import defaultdict
from threading import RLock
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .text import (
    DOMAIN_HINTS,
    QUERY_INTENTS,
    StoredChunk,
    StoredDocument,
    build_hybrid_query,
    build_snippet,
    expand_query,
    extract_document_chunks,
    normalize_text,
    query_contains_trigger,
    tokenize,
)


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
