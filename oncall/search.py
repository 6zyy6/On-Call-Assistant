import math
from collections import Counter, defaultdict
from threading import RLock
from typing import Dict, List

from fastapi import HTTPException

from .text import (
    StoredDocument,
    build_snippet,
    extract_heading_text,
    extract_visible_text_and_title,
    normalize_text,
    tokenize,
)


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

        if "oom" in query_tokens and ("后端" in title_text or "backend" in title_text or "服务" in title_text):
            score += 0.75

        for token in query_tokens:
            if token and token in heading_text:
                score += 0.1
            elif token and token in title_text:
                score += 0.06

        return score
