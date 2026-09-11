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


def test_as_of_recall_provider_configuration(monkeypatch):
    monkeypatch.setenv("CAURA_API_KEY", "dummy_key")
    from longmemeval.providers import get_memory_provider
    from longmemeval.providers.caura import CauraMemoryProvider

    p_on = get_memory_provider("caura", as_of_recall=True)
    assert isinstance(p_on, CauraMemoryProvider)
    assert p_on.send_valid_at is True

    p_off = get_memory_provider("caura", as_of_recall=False)
    assert isinstance(p_off, CauraMemoryProvider)
    assert p_off.send_valid_at is False


def test_split_context_windows_respects_chunk_boundaries():
    from longmemeval.llm import _split_context_windows

    chunks = [f"chunk-{i} " + ("x" * 1000) for i in range(10)]
    context = "\n---\n".join(chunks)
    windows = _split_context_windows(context, window_chars=3500)
    assert len(windows) == 4  # 3 chunks per window (3*~1010 < 3500), last window has 1
    # No chunk is split across windows
    rejoined = "\n---\n".join(windows)
    assert rejoined == context
    for w in windows:
        assert w.startswith("chunk-")


class WindowAwareLLM(BaseLLM):
    """Returns different facts depending on which window it sees; first call for window A is empty."""

    def __init__(self) -> None:
        self.calls = 0
        self.window_a_calls = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        if "EVENT-FEB-14" in prompt and "EVENT-FEB-15" not in prompt:
            return json.dumps({"status": "direct", "facts": ["Feb 14: 24-Hour Bike Ride charity event."], "requirements": ["dates"]})
        if "EVENT-FEB-15" in prompt:
            self.window_a_calls += 1
            if self.window_a_calls == 1:
                # Simulate the observed failure: valid JSON, status claims support, but no facts
                return json.dumps({"status": "inferable", "facts": [], "requirements": []})
            return json.dumps({"status": "inferable", "facts": ["Feb 15: Books for Kids charity book drive.", "Feb 14: 24-Hour Bike Ride charity event."], "requirements": ["dates"]})
        return json.dumps({"status": "unsupported", "facts": [], "requirements": []})


def test_extract_evidence_map_reduce_merges_windows_and_retries_empty():
    from longmemeval.llm import MAP_REDUCE_THRESHOLD_CHARS

    filler = "y" * 60_000
    chunk_a = "EVENT-FEB-14 " + filler
    chunk_b = "EVENT-FEB-15 " + filler
    chunk_c = "nothing relevant " + filler
    context = "\n---\n".join([chunk_a, chunk_b, chunk_c])
    assert len(context) > MAP_REDUCE_THRESHOLD_CHARS

    llm = WindowAwareLLM()
    bundle = llm.extract_evidence("How many months since two charity events on consecutive days?", context, "2023/04/18")

    assert bundle.status == "direct"  # best supported window with facts wins
    assert any("Feb 15" in f for f in bundle.facts)
    assert any("Feb 14" in f for f in bundle.facts)
    # Duplicate Feb 14 fact across windows is deduped
    assert sum("Feb 14" in f for f in bundle.facts) == 1
    # Window B was retried once after the empty-facts response
    assert llm.window_a_calls == 2


def test_map_reduce_drops_meta_negative_facts_and_passes_window_hint():
    from longmemeval.llm import _is_meta_negative

    assert _is_meta_negative("The retrieved chat history does not mention three completed road trips.")
    assert _is_meta_negative("No record of Bandung or Cihampelas Walk in the history.")
    assert not _is_meta_negative("On May 26, 2023, the user drove six hours to Washington D.C.")

    class NoisyLLM(BaseLLM):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kw):
            self.prompts.append(prompt)
            if "RELEVANT-CHUNK" in prompt:
                return json.dumps({"status": "direct", "facts": ["The assistant recommended Miss Bee Providore."], "requirements": ["name"]})
            return json.dumps({"status": "unsupported", "facts": ["The chat history does not mention any restaurant in Bandung."], "requirements": []})

    filler = "z" * 70_000
    context = "\n---\n".join(["irrelevant " + filler, "RELEVANT-CHUNK " + filler])
    llm = NoisyLLM()
    bundle = llm.extract_evidence("Remind me of the restaurant name?", context)
    assert bundle.facts == ("The assistant recommended Miss Bee Providore.",)
    assert bundle.status == "direct"
    assert all("window 1 of 2" in p or "window 2 of 2" in p for p in llm.prompts)


def test_extract_evidence_small_context_single_pass():
    llm = DummyLLM(responses={"Select and extract factual evidence": json.dumps({"status": "direct", "facts": ["f1"], "requirements": []})})
    bundle = llm.extract_evidence("q", "short context")
    assert bundle.facts == ("f1",)
    assert len(llm.call_history) == 1

