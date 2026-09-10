"""Provider registry."""

from __future__ import annotations

from .base import BaseMemoryProvider
from .caura import CauraMemoryProvider
from .baselines import OracleMemoryProvider, KeywordBM25MemoryProvider

PROVIDERS: dict[str, type[BaseMemoryProvider]] = {
    "caura": CauraMemoryProvider,
    "oracle": OracleMemoryProvider,
    "bm25": KeywordBM25MemoryProvider,
}


def get_memory_provider(name: str, **kwargs) -> BaseMemoryProvider:
    name_lower = name.lower()
    if name_lower not in PROVIDERS:
        raise ValueError(f"Unknown memory provider '{name}'. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name_lower](**kwargs)
