"""Tests for HTML report generation in LongMemEval."""

import json
from pathlib import Path
from longmemeval.report import (
    compute_quantiles,
    parse_context_chunks,
    build_report_payload,
    render_report,
    generate_report_for_run,
    secondary_eval_filename,
    summarize_secondary,
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


def _eval(results: list[tuple[str, str, bool]], judge: str) -> dict:
    by_type: dict[str, dict] = {}
    for qid, qt, ok in results:
        d = by_type.setdefault(qt, {"accuracy": 0.0, "total": 0, "correct": 0})
        d["total"] += 1
        d["correct"] += int(ok)
    for d in by_type.values():
        d["accuracy"] = d["correct"] / d["total"]
    return {
        "overall_accuracy": sum(ok for *_, ok in results) / len(results),
        "total_questions": len(results),
        "correct_questions": sum(ok for *_, ok in results),
        "by_question_type": by_type,
        "results": [
            {"question_id": q, "question": "q", "gold_answer": "g", "hypothesis": "h", "question_type": qt,
             "correct": ok, "judge_reason": "yes" if ok else "no", "judge_model": judge}
            for q, qt, ok in results
        ],
    }


def test_secondary_eval_filename_slug():
    assert secondary_eval_filename("gemini-3.5-flash-lite") == "eval_results_gemini35flashlite.json"
    assert secondary_eval_filename("gpt-4o") == "eval_results_gpt4o.json"
    assert secondary_eval_filename("GPT-5.6 Terra") == "eval_results_gpt56terra.json"


def test_summarize_secondary_agreement_and_flips():
    primary = _eval([("a", "t", True), ("b", "t", True), ("c", "m", False), ("d", "m", False)], "gpt-4o")
    secondary = _eval([("a", "t", True), ("b", "t", False), ("c", "m", True), ("d", "m", False)], "flash-lite")
    block = summarize_secondary("flash-lite", "eval_results_flashlite.json", primary, secondary)
    assert block["judge"] == "flash-lite"
    assert block["correct_questions"] == 2
    assert block["agreement_with_primary"] == 2
    assert block["primary_only_correct"] == ["b"]
    assert block["secondary_only_correct"] == ["c"]
    assert block["by_question_type"]["t"]["correct"] == 1


def test_generate_report_carries_secondary_judge(tmp_path: Path):
    run_dir = tmp_path / "dual-judge-run"
    run_dir.mkdir()
    primary = _eval([("a", "temporal-reasoning", True), ("b", "multi-session", False)], "gpt-4o")
    secondary = _eval([("a", "temporal-reasoning", False), ("b", "multi-session", False)], "gemini-3.5-flash-lite")
    (run_dir / "eval_results.json").write_text(json.dumps(primary), encoding="utf-8")
    (run_dir / "eval_results_gemini35flashlite.json").write_text(json.dumps(secondary), encoding="utf-8")
    (run_dir / "hypotheses.jsonl").write_text(
        "\n".join(json.dumps({"question_id": q, "hypothesis": "h", "context": "c", "retrieve_time_ms": 1, "generate_time_ms": 1}) for q in "ab") + "\n",
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps({"run": {"name": "dual-judge-run", "judge": "gpt-4o", "judge_secondary": "gemini-3.5-flash-lite"}}),
        encoding="utf-8",
    )

    generate_report_for_run(run_dir)

    saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    sec = saved["secondary_evaluation"]
    assert sec["judge"] == "gemini-3.5-flash-lite"
    assert sec["file"] == "eval_results_gemini35flashlite.json"
    assert sec["correct_questions"] == 0
    assert sec["agreement_with_primary"] == 1
    # Headline stays the primary judge's number.
    assert saved["summary"]["correct_questions"] == 1
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert '"secondary_evaluation"' in html
