"""In-memory baseline providers: Oracle (gold sessions only) and Simple BM25/keyword."""

from __future__ import annotations

import re
from ..models import MemoryDocument, RetrievedFact
from .base import BaseMemoryProvider


class OracleMemoryProvider(BaseMemoryProvider):
    """Oracle memory provider: retrieves only gold answer turns/sessions."""

    name = "oracle"

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
        # Return all documents stored for this unit
        docs = self._store.get(unit_id, [])
        return [
            RetrievedFact(
                id=d.id,
                content=d.content,
                score=1.0,
                timestamp=d.timestamp,
            )
            for d in docs[:top_k]
        ]


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
