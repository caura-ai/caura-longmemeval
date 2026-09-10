"""Tests for HTML report generation in LongMemEval."""

import json
from pathlib import Path
from longmemeval.report import (
    compute_quantiles,
    parse_context_chunks,
    build_report_payload,
    render_report,
    generate_report_for_run,
)


def test_compute_quantiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    q = compute_quantiles(data)
    assert q["min"] == 10.0
    assert q["max"] == 50.0
    assert q["avg"] == 30.0
    assert q["p50"] == 30.0
    assert q["p95"] >= 45.0

    empty = compute_quantiles([])
    assert empty["avg"] == 0.0


def test_parse_context_chunks():
    raw_context = (
        "[date: 2023-04-29T00:02:00+00:00 | context: Session s1 | part 1/2]\n"
        "User: Hello world\n"
        "Assistant: Hi!\n"
        "---\n"
        "[date: 2023-04-30T10:00:00+00:00 | context: Session s2]\n"
        "User: Fact two\n"
    )
    chunks = parse_context_chunks(raw_context, max_chunks=5, preview_chars=100)
    assert len(chunks) == 2
    assert chunks[0]["rank"] == 1
    assert chunks[0]["date"] == "2023-04-29T00:02:00+00:00"
    assert "Hello world" in chunks[0]["preview"]
    assert chunks[1]["rank"] == 2
    assert chunks[1]["date"] == "2023-04-30T10:00:00+00:00"


def test_build_report_payload_and_render(tmp_path: Path):
    run_meta = {
        "name": "test-run",
        "provider": "caura",
        "started_at": "2026-09-10T12:00:00Z",
        "finished_at": "2026-09-10T12:05:00Z",
        "duration_seconds": 300.0,
        "reader": "gemini-3.8-flash",
        "judge": "gemini-3.5-flash-lite",
        "top_k": 20,
    }
    summary = {
        "overall_accuracy": 0.8,
        "total_questions": 5,
        "correct_questions": 4,
        "by_question_type": {
            "single-session-user": {"accuracy": 1.0, "total": 2, "correct": 2},
            "multi-session": {"accuracy": 0.6666, "total": 3, "correct": 2},
        },
        "results": [
            {
                "question_id": "q1",
                "question": "What is my cat's name?",
                "gold_answer": "Milo",
                "hypothesis": "Your cat is Milo.",
                "question_type": "single-session-user",
                "correct": True,
                "judge_reason": "yes",
                "judge_model": "gemini-3.5-flash-lite",
            },
            {
                "question_id": "q2_abs",
                "question": "When did I buy a helicopter?",
                "gold_answer": "",
                "hypothesis": "You never mentioned buying a helicopter.",
                "question_type": "multi-session",
                "correct": True,
                "judge_reason": "yes",
                "judge_model": "gemini-3.5-flash-lite",
            },
        ],
    }
    hypotheses = [
        {
            "question_id": "q1",
            "hypothesis": "Your cat is Milo.",
            "retrieve_time_ms": 1200.0,
            "generate_time_ms": 1800.0,
            "context": "[date: 2024-01-01] Cat name is Milo",
        },
        {
            "question_id": "q2_abs",
            "hypothesis": "You never mentioned buying a helicopter.",
            "retrieve_time_ms": 1500.0,
            "generate_time_ms": 2100.0,
            "context": "[date: 2024-01-02] No helicopters",
        },
    ]

    payload = build_report_payload(run_meta, summary, hypotheses)

    assert payload["benchmark"] == "LongMemEval"
    assert payload["summary"]["overall_accuracy"] == 0.8
    assert payload["summary"]["abstention_count"] == 1
    assert payload["summary"]["abstention_accuracy"] == 1.0
    assert len(payload["questions"]) == 2
    assert payload["summary"]["retrieval"]["avg"] > 0

    out_file = tmp_path / "report.html"
    rendered = render_report(payload, out_file)

    assert rendered.exists()
    content = rendered.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "Caura × LongMemEval — test-run" in content
    assert "Milo" in content
    assert "__DATA__" not in content


def test_generate_report_for_existing_run(tmp_path: Path):
    run_dir = tmp_path / "mock-run"
    run_dir.mkdir()

    eval_data = {
        "overall_accuracy": 1.0,
        "total_questions": 1,
        "correct_questions": 1,
        "by_question_type": {
            "temporal-reasoning": {"accuracy": 1.0, "total": 1, "correct": 1}
        },
        "results": [
            {
                "question_id": "q_temp",
                "question": "How many days between events?",
                "gold_answer": "5 days",
                "hypothesis": "5 days",
                "question_type": "temporal-reasoning",
                "correct": True,
                "judge_reason": "yes",
                "judge_model": "gemini-3.5-flash-lite",
            }
        ],
    }
    (run_dir / "eval_results.json").write_text(json.dumps(eval_data), encoding="utf-8")
    (run_dir / "hypotheses.jsonl").write_text(
        json.dumps({
            "question_id": "q_temp",
            "hypothesis": "5 days",
            "retrieve_time_ms": 500,
            "generate_time_ms": 1000,
            "context": "event1 ... event2",
        }) + "\n",
        encoding="utf-8",
    )

    out_report = generate_report_for_run(run_dir)
    assert out_report.exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "report.html").exists()
