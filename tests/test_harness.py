"""Tests for LongMemEval harness."""

import pytest
from longmemeval.models import LongMemEvalItem, MemoryDocument
from longmemeval.dataset import LongMemEvalDataset
from longmemeval.providers.baselines import OracleMemoryProvider, KeywordBM25MemoryProvider
from longmemeval.prompts import build_answer_prompt, get_official_judge_prompt


def test_models():
    item = LongMemEvalItem(
        question_id="test_q1",
        question="Where did I travel in July?",
        answer="Paris",
        question_type="single-session-user",
        question_date="2024/08/01",
        haystack_sessions=[
            [{"role": "user", "content": "I visited Paris in July."}]
        ],
        haystack_dates=["2024/07/15"],
        haystack_session_ids=["s1"],
    )
    assert item.question_id == "test_q1"

    ds = LongMemEvalDataset()
    docs = ds.item_to_documents(item)
    assert len(docs) == 1
    assert "Paris" in docs[0].content


def test_documents_carry_no_dataset_labels():
    """Gold sessions are named answer_<hash> and abstention questions <id>_abs in the
    dataset; neither may reach the stored text or its header."""
    from longmemeval.dataset import opaque_session_id

    item = LongMemEvalItem(
        question_id="abc123_abs",
        question="What is my cat called?",
        answer="N/A",
        question_type="single-session-user",
        question_date="2024/08/01",
        haystack_sessions=[
            [{"role": "user", "content": "I adopted a dog.", "has_answer": True}],
            [{"role": "user", "content": "Weather is nice."}],
        ],
        haystack_dates=["2024/07/15 (Mon) 10:00", "2024/07/16 (Tue) 10:00"],
        haystack_session_ids=["answer_deadbeef", "sharegpt_x_0"],
    )
    docs = LongMemEvalDataset().item_to_documents(item)
    labels = [opaque_session_id(item.question_id, sid) for sid in item.haystack_session_ids]
    for doc, label in zip(docs, labels):
        assert doc.id == f"abc123_abs_{label}"  # bookkeeping id keeps the question id
        assert doc.context == f"Session {label} - happened on {doc.timestamp[:10]} {doc.timestamp[11:19]} UTC."
        for leak in ("answer_", "deadbeef", "sharegpt", "_abs", "abc123"):
            assert leak not in doc.context
            assert leak not in doc.content
    # Same shape for gold and distractor labels; deterministic across calls.
    assert len(set(labels)) == 2 and all(len(l) == 12 for l in labels)
    assert opaque_session_id("q", "s") == opaque_session_id("q", "s")


def test_baseline_providers():
    from longmemeval.providers.baselines import FullContextMemoryProvider

    docs = [
        MemoryDocument(id="d1", content="Fact 1", user_id="u1", is_gold=True),
        MemoryDocument(id="d2", content="Fact 2", user_id="u1"),
    ]
    # Oracle: gold sessions only, top_k ignored.
    oracle = OracleMemoryProvider()
    assert oracle.ingest("u1", docs) == 1
    ret = oracle.retrieve("u1", "Fact 1", top_k=1)
    assert [r.id for r in ret] == ["d1"]

    # Full context: everything, top_k ignored.
    full = FullContextMemoryProvider()
    assert full.ingest("u1", docs) == 2
    assert [r.id for r in full.retrieve("u1", "anything", top_k=1)] == ["d1", "d2"]

    bm25 = KeywordBM25MemoryProvider()
    bm25.ingest("u1", docs)
    ret_bm25 = bm25.retrieve("u1", "Fact 2")
    assert len(ret_bm25) == 2
    assert ret_bm25[0].content == "Fact 2"


def test_prompts():
    prompt = build_answer_prompt("What is my name?", "User's name is Alice", "2024/01/01")
    assert "What is my name?" in prompt
    assert "Alice" in prompt

    judge_p = get_official_judge_prompt(
        task="knowledge-update",
        question="What is my job?",
        answer="Doctor",
        response="She works as a doctor now.",
    )
    assert "knowledge-update" not in judge_p  # Template itself
    assert "Doctor" in judge_p
    assert "Answer yes or no only" in judge_p


def test_reader_prompts_contain_no_dataset_examples():
    """The reader prompts were iterated against failures on this set; the examples
    they carry must be generic or synthetic, never lifted from LongMemEval items."""
    from longmemeval.prompts import (
        build_evidence_answer_prompt,
        build_extract_evidence_prompt,
        build_infer_answer_prompt,
        build_verify_answer_prompt,
    )

    templates = " ".join([
        build_extract_evidence_prompt("Q", "CTX", "D"),
        build_evidence_answer_prompt("Q", "EV", "D"),
        build_infer_answer_prompt("Q", "EV", "D"),
        build_verify_answer_prompt("Q", "EV", ("C1",), "D"),
    ]).lower()
    # Terms that were lifted from specific test questions in an earlier version.
    lifted = [
        "slow cooker", "beef stew", "yogurt", "mortgage", "pre-approval", "feb 14", "feb 15",
        "consecutive days", "chicago", "suburbs", "suggest a hotel", "target,", "cartwheel",
        "healthcare ai", "avoid screens", "led 2 projects", "never classify recommendation",
    ]
    hits = [t for t in lifted if t in templates]
    assert not hits, f"dataset-lifted wording in reader prompts: {hits}"


def test_caura_category_adaptive_profiles():
    from longmemeval.providers.caura import CauraMemoryProvider, CATEGORY_SEARCH_PROFILES, MAX_SEARCH_TOP_K

    assert MAX_SEARCH_TOP_K == 200
    assert "temporal-reasoning" in CATEGORY_SEARCH_PROFILES
    assert CATEGORY_SEARCH_PROFILES["temporal-reasoning"]["top_k"] == 50
    assert CATEGORY_SEARCH_PROFILES["multi-session"]["top_k"] == 60
    assert CATEGORY_SEARCH_PROFILES["single-session-preference"]["top_k"] == 15
    assert CATEGORY_SEARCH_PROFILES["knowledge-update"]["top_k"] == 30

    provider = CauraMemoryProvider(api_key="test_key", tenant_id="test_tenant")
    assert provider.category_adaptive is True
    assert provider.chunk_chars == 4000
    assert provider.chunk_mode == "chars"
    assert provider.sibling_expansion is False
    assert (provider.top_k, provider.multiquery, provider.merge_top_k) == (20, 2, 35)


def test_turn_mode_defaults_to_flat_profile_with_sibling_expansion(monkeypatch):
    from longmemeval.providers.caura import CauraMemoryProvider

    # Legacy .env values for the 4k store must not leak into turn mode.
    monkeypatch.setenv("CAURA_TOP_K", "20")
    monkeypatch.setenv("CAURA_MULTIQUERY", "2")
    monkeypatch.setenv("CAURA_CATEGORY_ADAPTIVE", "true")
    monkeypatch.setenv("CAURA_CHUNK_CHARS", "4000")

    p = CauraMemoryProvider(api_key="k", tenant_id="t", chunk_mode="turns")
    assert p.chunk_chars == 1200
    assert p.category_adaptive is False
    assert (p.top_k, p.multiquery, p.merge_top_k) == (50, 1, 50)
    assert p.sibling_expansion is True
    assert p.sibling_window == 0

    # Explicit args still win.
    p2 = CauraMemoryProvider(api_key="k", tenant_id="t", chunk_mode="turns", top_k=30, multiquery=2, category_adaptive=True)
    assert (p2.top_k, p2.multiquery, p2.category_adaptive) == (30, 2, True)


def test_chunk_turns_groups_user_turn_with_reply_and_splits_long_replies():
    from longmemeval.providers.caura import _chunk_turns

    short_reply = "Assistant: Sure, noted."
    long_reply = "Assistant: " + " ".join(f"sentence {i}." for i in range(400))  # ~4.5k chars
    text = "\n\n".join([
        "User: I adopted a puppy named Biscuit last week.",
        short_reply,
        "User: Can you suggest a training schedule?",
        long_reply,
        "User: Thanks!",
    ])
    chunks = _chunk_turns(text, size=1200)

    # First exchange fits: user + reply in one chunk.
    assert chunks[0].startswith("User: I adopted a puppy named Biscuit")
    assert short_reply in chunks[0]
    # Second exchange: user turn alone, then reply pieces each anchored to the user turn.
    assert chunks[1] == "User: Can you suggest a training schedule?"
    reply_pieces = [c for c in chunks if c.startswith("(in reply to) User: Can you suggest a training schedule?")]
    assert len(reply_pieces) >= 4
    assert all(len(c) <= 1200 for c in chunks)
    # Trailing user turn survives as its own memory.
    assert chunks[-1] == "User: Thanks!"
    # Nothing lost.
    assert "sentence 399." in "".join(chunks)


def test_sibling_expansion_completes_best_ranked_sessions_within_budget(monkeypatch):
    from longmemeval.providers.caura import CauraMemoryProvider

    provider = CauraMemoryProvider(
        api_key="k", tenant_id="t", chunk_mode="turns", sibling_expansion=True,
        context_budget_chars=1000, sibling_window=0,
    )
    assert provider.chunk_chars == 1200  # turns mode ignores CAURA_CHUNK_CHARS

    def mem(mid, doc, chunk, ts, size=100):
        return {
            "id": mid,
            "content": "x" * size,
            "ts_valid_start": ts,
            "metadata": {"doc_id": doc, "chunk": chunk},
        }

    # Store: session A (3 chunks, older), session B (3 chunks, newer), session C (2 chunks).
    store = [
        mem("a0", "q1_A", 0, "2023-01-01"), mem("a1", "q1_A", 1, "2023-01-01"), mem("a2", "q1_A", 2, "2023-01-01"),
        mem("b0", "q1_B", 0, "2023-02-01"), mem("b1", "q1_B", 1, "2023-02-01"), mem("b2", "q1_B", 2, "2023-02-01"),
        mem("c0", "q1_C", 0, "2023-03-01"), mem("c1", "q1_C", 1, "2023-03-01"),
    ]
    monkeypatch.setattr(provider, "_list_agent_memories", lambda agent_id, unit_id: list(store))

    # Ranked hits: one chunk from B (best), one from A, one from C.
    ranked = [store[4], store[0], store[7]]
    selected = provider._expand_siblings(ranked, "lme-q1", "q1", seed_limit=10, valid_at=None)
    ids = [s["id"] for s in selected]

    # Budget 1000 chars = 10 chunks of 100, store has 8 -> everything fits.
    assert set(ids) == {m["id"] for m in store}
    # Ordered by session date then chunk index, so each session is contiguous and in order.
    assert ids == ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1"]

    # Tight budget: best-ranked hit's session (B) is completed first, then A partially; C never reached.
    provider.context_budget_chars = 500
    selected = provider._expand_siblings(ranked, "lme-q1", "q1", seed_limit=10, valid_at=None)
    ids = set(s["id"] for s in selected)
    assert ids == {"b0", "b1", "b2", "a0", "a1"}

    # Window of 1 around each hit: session A hit at chunk 0 -> only a1 is pulled, not a2.
    provider.context_budget_chars = 10_000
    provider.sibling_window = 1
    selected = provider._expand_siblings(ranked, "lme-q1", "q1", seed_limit=10, valid_at=None)
    ids = set(s["id"] for s in selected)
    assert "a1" in ids and "a2" not in ids
    assert {"b0", "b1", "b2", "c0", "c1"} <= ids


def test_raw_turns_only_drops_server_derived_search_hits(monkeypatch):
    from longmemeval.providers.caura import CauraMemoryProvider

    provider = CauraMemoryProvider(
        api_key="k", tenant_id="t", chunk_mode="turns", sibling_expansion=False, send_valid_at=False,
    )
    assert provider.raw_turns_only is True  # default in turn mode

    stored = {"id": "m1", "content": "User: I adopted a dog.", "metadata": {"doc_id": "q1_abc", "chunk": 0}}
    derived = {"id": "d1", "content": "User owns a dog.", "memory_type": "fact", "metadata": {}}
    monkeypatch.setattr(provider, "_search_once", lambda *a, **k: [derived, stored, derived | {"id": "d2"}])

    facts = provider.retrieve("q1", "what pet do I have?")
    assert [f.id for f in facts] == ["m1"]
    assert provider.pop_retrieval_stats() == {
        "search_candidates": 3, "server_derived_candidates": 2, "server_derived_dropped": 2,
    }
    assert provider.pop_retrieval_stats() == {}  # consumed

    provider.raw_turns_only = False
    facts = provider.retrieve("q1", "what pet do I have?")
    assert [f.id for f in facts] == ["d1", "m1", "d2"]
    assert provider.pop_retrieval_stats()["server_derived_dropped"] == 0


def test_judge_protocol_defaults_and_overrides(monkeypatch):
    from longmemeval.cli import resolve_judges

    for k in ("JUDGE_LLM", "JUDGE_MODEL", "SECONDARY_JUDGE_LLM", "SECONDARY_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)

    # Protocol default: gpt-4o headline, flash-lite secondary.
    primary, secondary = resolve_judges(None, None, None, None, no_secondary=False)
    assert primary == ("openai", "gpt-4o")
    assert secondary == ("gemini", "gemini-3.5-flash-lite")

    # --no-secondary-judge drops the secondary only.
    assert resolve_judges(None, None, None, None, no_secondary=True) == (("openai", "gpt-4o"), None)

    # Env pins both slots (what .env does); CLI flags beat env.
    monkeypatch.setenv("JUDGE_LLM", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
    monkeypatch.setenv("SECONDARY_JUDGE_LLM", "gemini")
    monkeypatch.setenv("SECONDARY_JUDGE_MODEL", "gemini-3.5-flash-lite")
    primary, secondary = resolve_judges("openai", "gpt-5.6-terra", None, None, no_secondary=False)
    assert primary == ("openai", "gpt-5.6-terra")
    assert secondary == ("gemini", "gemini-3.5-flash-lite")

    # A judge study that makes primary == secondary collapses to a single judge instead of judging twice.
    primary, secondary = resolve_judges("gemini", "gemini-3.5-flash-lite", None, None, no_secondary=False)
    assert primary == ("gemini", "gemini-3.5-flash-lite") and secondary is None

    # Provider switched without a model -> provider default, not the other slot's model.
    monkeypatch.delenv("JUDGE_MODEL")
    primary, _ = resolve_judges("gemini", None, None, None, no_secondary=True)
    assert primary == ("gemini", None)
