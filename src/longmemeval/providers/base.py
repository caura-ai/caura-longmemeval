"""Memory provider interface for LongMemEval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from ..models import MemoryDocument, RetrievedFact


class BaseMemoryProvider(ABC):
    """Abstract interface for LongMemEval memory providers."""

    name: str

    @abstractmethod
    def reset_unit(self, unit_id: str) -> None:
        """Purge any memories for this specific isolation unit (question_id)."""
        pass

    @abstractmethod
    def ingest(self, unit_id: str, documents: list[MemoryDocument]) -> int:
        """Ingest conversation documents for the given unit_id. Returns stored count."""
        pass

    @abstractmethod
    def retrieve(
        self,
        unit_id: str,
        query: str,
        top_k: int = 20,
        query_date: str | None = None,
        question_type: str | None = None,
    ) -> list[RetrievedFact]:
        """Retrieve relevant memories for the unit and query."""
        pass

    def cleanup(self, unit_id: str | None = None) -> None:
        """Optional provider cleanup at end of run or per-unit."""
        pass
