"""In-memory baseline providers: Oracle (gold sessions only) and Simple BM25/keyword."""

from __future__ import annotations

import re
from ..models import MemoryDocument, RetrievedFact
from .base import BaseMemoryProvider


def _as_facts(docs: list[MemoryDocument]) -> list[RetrievedFact]:
    return [
        RetrievedFact(
            id=d.id,
            content=(f"{d.context}\n{d.content}" if d.context else d.content),
            score=1.0,
            timestamp=d.timestamp,
        )
        for d in docs
    ]


class OracleMemoryProvider(BaseMemoryProvider):
    """Oracle baseline: the reader sees exactly the question's answer sessions
    (``MemoryDocument.is_gold``), whole, in chronological order. Abstention
    questions have no gold session and get an empty context. ``top_k`` is
    ignored; this is the reader's ceiling given perfect retrieval."""

    name = "oracle"

    def __init__(self):
        self._store: dict[str, list[MemoryDocument]] = {}

    def reset_unit(self, unit_id: str) -> None:
        self._store.pop(unit_id, None)

    def ingest(self, unit_id: str, documents: list[MemoryDocument]) -> int:
        gold = [d for d in documents if d.is_gold]
        self._store[unit_id] = gold
        return len(gold)

    def retrieve(
        self,
        unit_id: str,
        query: str,
        top_k: int = 20,
        query_date: str | None = None,
        question_type: str | None = None,
    ) -> list[RetrievedFact]:
        return _as_facts(self._store.get(unit_id, []))


class FullContextMemoryProvider(BaseMemoryProvider):
    """Full-context baseline: no retrieval at all, the reader sees the whole
    haystack (every session, chronological). ``top_k`` is ignored."""

    name = "fullcontext"

    def __init__(self):
        self._store: dict[str, list[MemoryDocument]] = {}

    def reset_unit(self, unit_id: str) -> None:
        self._store.pop(unit_id, None)

    def ingest(self, unit_id: str, documents: list[MemoryDocument]) -> int:
        self._store[unit_id] = list(documents)
        return len(documents)

    def retrieve(
        self,
        unit_id: str,
        query: str,
        top_k: int = 20,
        query_date: str | None = None,
        question_type: str | None = None,
    ) -> list[RetrievedFact]:
        return _as_facts(self._store.get(unit_id, []))


class KeywordBM25MemoryProvider(BaseMemoryProvider):
    """Simple term-frequency retriever for baseline comparison without external services."""

    name = "bm25"

    def __init__(self):
        self._store: dict[str, list[MemoryDocument]] = {}

    def reset_unit(self, unit_id: str) -> None:
        self._store.pop(unit_id, None)

    def ingest(self, unit_id: str, documents: list[MemoryDocument]) -> int:
        self._store[unit_id] = documents
        return len(documents)

    def retrieve(
        self,
        unit_id: str,
        query: str,
        top_k: int = 20,
        query_date: str | None = None,
        question_type: str | None = None,
    ) -> list[RetrievedFact]:
        docs = self._store.get(unit_id, [])
        if not docs:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, MemoryDocument]] = []
        for d in docs:
            doc_tokens = re.findall(r"\w+", d.content.lower())
            score = sum(1 for t in doc_tokens if t in query_tokens)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedFact(
                id=d.id,
                content=d.content,
                score=s,
                timestamp=d.timestamp,
            )
            for s, d in scored[:top_k]
        ]
