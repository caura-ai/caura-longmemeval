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
    assert docs[0].id == "test_q1_s1"
    assert "Paris" in docs[0].content


def test_baseline_providers():
    oracle = OracleMemoryProvider()
    docs = [
        MemoryDocument(id="d1", content="Fact 1", user_id="u1"),
        MemoryDocument(id="d2", content="Fact 2", user_id="u1"),
    ]
    stored = oracle.ingest("u1", docs)
    assert stored == 2

    ret = oracle.retrieve("u1", "Fact 1")
    assert len(ret) == 2

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


def test_caura_category_adaptive_profiles():
    from longmemeval.providers.caura import CauraMemoryProvider, CATEGORY_SEARCH_PROFILES, MAX_SEARCH_TOP_K

    assert MAX_SEARCH_TOP_K == 200
    assert "temporal-reasoning" in CATEGORY_SEARCH_PROFILES
    assert CATEGORY_SEARCH_PROFILES["temporal-reasoning"]["top_k"] == 50
    assert CATEGORY_SEARCH_PROFILES["multi-session"]["top_k"] == 60
    assert CATEGORY_SEARCH_PROFILES["single-session-preference"]["top_k"] == 15

    provider = CauraMemoryProvider(api_key="test_key", tenant_id="test_tenant")
    assert provider.category_adaptive is True
    assert provider.chunk_chars == 4000
