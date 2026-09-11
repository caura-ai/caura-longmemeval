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
    cls = PROVIDERS[name_lower]
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    }
    return cls(**valid_args)
