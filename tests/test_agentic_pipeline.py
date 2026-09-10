"""Unit tests for agentic-v1 multi-stage pipeline and holdout exclusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from longmemeval.dataset import LongMemEvalDataset, extract_question_ids_from_file
from longmemeval.llm import BaseLLM, _parse_json_object
from longmemeval.models import LongMemEvalItem, EvidenceBundle, VerifiedAnswer
from longmemeval.runner import run_reader_pipeline


class DummyLLM(BaseLLM):
    """Deterministic dummy LLM for testing multi-stage calls."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.model = "dummy-model"
        self.responses = responses or {}
        self.call_history: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_history.append(prompt)
        for key, resp in self.responses.items():
            if key in prompt:
                return resp
        return "dummy answer"


def test_parse_json_object():
    # Plain JSON
    raw = '{"status": "direct", "facts": ["fact 1", "fact 2"]}'
    parsed = _parse_json_object(raw)
    assert parsed.get("status") == "direct"
    assert len(parsed.get("facts", [])) == 2

    # Markdown fence
    fence = '```json\n{"status": "inferable", "requirements": ["req 1"]}\n```'
    parsed2 = _parse_json_object(fence)
    assert parsed2.get("status") == "inferable"

    # Preceding/trailing conversational text
    noisy = 'Here is the JSON you requested:\n{"answer": "Paris", "reason": "Verified"}\nHope this helps!'
    parsed3 = _parse_json_object(noisy)
    assert parsed3.get("answer") == "Paris"

    # Invalid JSON returns None
    assert _parse_json_object("invalid non-json text") is None


def test_dummy_llm_agentic_stages():
    dummy_responses = {
        "Select and extract factual evidence": json.dumps({
            "status": "direct",
            "facts": ["2023-01-10: User bought a red bike.", "2023-05-12: User replaced tires."],
            "requirements": ["Color of the bike"],
        }),
        "Answer the question based strictly on the extracted conversational evidence": "The bike is red.",
        "Verify and, when necessary, correct the candidate answer": json.dumps({
            "answer": "The user purchased a red bike.",
            "reason": "Directly supported by 2023-01-10 fact.",
        }),
    }
    llm = DummyLLM(responses=dummy_responses)

    # Test extract_evidence
    bundle = llm.extract_evidence("What color is the bike?", "2023-01-10: bought red bike")
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.status == "direct"
    assert len(bundle.facts) == 2
    assert "User bought a red bike." in bundle.facts[0]

    # Test answer_from_evidence
    ans = llm.answer_from_evidence("What color is the bike?", bundle)
    assert ans == "The bike is red."

    # Test verify_answer
    ver = llm.verify_answer("What color is the bike?", bundle, candidates=(ans,))
    assert isinstance(ver, VerifiedAnswer)
    assert ver.answer == "The user purchased a red bike."
    assert "Directly supported" in ver.reason


def test_run_reader_pipeline_agentic_flow():
    dummy_responses = {
        "Select and extract factual evidence": json.dumps({
            "status": "direct",
            "facts": ["Fact A: The dog name is Barnaby."],
            "requirements": ["Dog name"],
        }),
        "Answer the question based strictly on the extracted conversational evidence": "Barnaby",
        "Verify and, when necessary, correct the candidate answer": json.dumps({
            "answer": "Barnaby",
            "reason": "Verified from Fact A.",
        }),
    }
    llm = DummyLLM(responses=dummy_responses)

    ans, gen_ms, trace = run_reader_pipeline(
        reader_llm=llm,
        question="What is the dog's name?",
        context="Fact A: The dog name is Barnaby.",
        pipeline="agentic-v1",
    )
    assert ans == "Barnaby"
    assert gen_ms >= 0
    assert trace is not None
    assert trace["pipeline"] == "agentic-v1"
    assert trace["evidence_status"] == "direct"
    assert "Fact A: The dog name is Barnaby." in trace["facts"]
    assert trace["verifier_reason"] == "Verified from Fact A."


def test_extract_question_ids_from_file(tmp_path: Path):
    eval_file = tmp_path / "eval_results.json"
    eval_file.write_text(json.dumps({
        "results": [
            {"question_id": "q1", "correct": True},
            {"question_id": "q2", "correct": False},
        ]
    }), encoding="utf-8")

    ids = extract_question_ids_from_file(eval_file)
    assert ids == {"q1", "q2"}

    # Hypotheses JSONL
    hypo_file = tmp_path / "hypotheses.jsonl"
    hypo_file.write_text(
        '{"question_id": "q3", "hypothesis": "a"}\n{"question_id": "q4", "hypothesis": "b"}\n',
        encoding="utf-8",
    )
    ids_hypo = extract_question_ids_from_file(hypo_file)
    assert ids_hypo == {"q3", "q4"}


def test_load_items_with_exclusion():
    items = [
        LongMemEvalItem(
            question_id=f"item_{i}",
            question=f"Question {i}",
            answer=f"Answer {i}",
            question_type="single-session-user",
            question_date=f"2023-01-{i+1:02d}",
            haystack_session_ids=[f"s_{i}"],
            haystack_sessions=[[{"role": "user", "content": f"msg {i}"}]],
            haystack_dates=[f"2023-01-{i+1:02d}"],
        )
        for i in range(10)
    ]
    ds = LongMemEvalDataset()
    ds._items = items

    # Exclude items 0, 1, 2
    filtered = ds.load_items(exclude_ids={"item_0", "item_1", "item_2"})
    assert len(filtered) == 7
    loaded_ids = {x.question_id for x in filtered}
    assert "item_0" not in loaded_ids
    assert "item_1" not in loaded_ids
    assert "item_2" not in loaded_ids
