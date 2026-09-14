"""LongMemEval benchmark models and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class EvidenceBundle:
    status: str  # "direct" | "inferable" | "unsupported"
    facts: tuple[str, ...]
    requirements: tuple[str, ...]
    windows: int = 1  # >1 when the windowed (map-reduce) fallback produced this bundle

    def as_context(self) -> str:
        lines = [f"Support status: {self.status}"]
        if self.requirements:
            lines.append("Requirements:")
            lines.extend(f"- {req}" for req in self.requirements)
        lines.append("Extracted facts:")
        if self.facts:
            lines.extend(f"- {fact}" for fact in self.facts)
        else:
            lines.append("- None")
        return "\n".join(lines)


@dataclass(frozen=True)
class VerifiedAnswer:
    answer: str
    reason: str


class ChatTurn(BaseModel):
    role: str
    content: str
    has_answer: bool = False


class LongMemEvalItem(BaseModel):
    question_id: str
    question: str
    answer: str = ""
    question_type: str
    question_date: str
    haystack_sessions: list[list[dict[str, Any]]]
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    other_attributes: dict[str, Any] = Field(default_factory=dict)


class MemoryDocument(BaseModel):
    id: str
    content: str
    user_id: str  # maps to question_id in LongMemEval for per-question isolation
    timestamp: str | None = None
    context: str | None = None
    # Dataset label: this session is one of the question's answer sessions. Used only
    # by the in-process oracle baseline; the Caura provider never reads or sends it.
    is_gold: bool = False


class RetrievedFact(BaseModel):
    id: str
    content: str
    score: float | None = None
    timestamp: str | None = None
    memory_type: str | None = None
    title: str | None = None
    tags: list[str] = Field(default_factory=list)


class HypothesisEntry(BaseModel):
    question_id: str
    hypothesis: str
    question: str | None = None
    answer: str | None = None
    question_type: str | None = None
    context: str | None = None
    retrieve_time_ms: float = 0.0
    generate_time_ms: float = 0.0
    pipeline: str | None = "direct"
    pipeline_trace: dict[str, Any] | None = None
    # Provider-side counters for this question (e.g. how many /search candidates
    # were server-derived memories and whether they were dropped).
    retrieval_stats: dict[str, Any] | None = None
    # Provider-reported reader token usage summed over every call for this question
    # (extract/answer/infer/verify, retries and fallbacks included). None if unreported.
    reader_usage: dict[str, Any] | None = None


class EvaluationResult(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    hypothesis: str
    question_type: str
    correct: bool
    judge_reason: str
    judge_model: str
