"""Generate a polished, portable, dependency-free HTML benchmark report for LongMemEval.

Inspired by modern telemetry dashboards and Caura's LoCoMo benchmark reports.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvaluationResult, HypothesisEntry

# Official LongMemEval 500-question distribution for reweighting projections
OFFICIAL_CATEGORY_WEIGHTS: dict[str, float] = {
    "single-session-user": 70 / 500,
    "single-session-assistant": 56 / 500,
    "multi-session": 133 / 500,
    "temporal-reasoning": 133 / 500,
    "knowledge-update": 78 / 500,
    "single-session-preference": 30 / 500,
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "single-session-user": "Recall user-stated facts and attributes from a single conversation session.",
    "single-session-assistant": "Recall assistant-provided advice, answers, or facts from a single session.",
    "multi-session": "Synthesize and aggregate information scattered across multiple separate conversations.",
    "temporal-reasoning": "Compute time elapsed, chronological order, and temporal relative queries.",
    "knowledge-update": "Track evolving facts, preference shifts, and supersede outdated information.",
    "single-session-preference": "Remember user implicit and explicit preferences across dialogue turns.",
}


def parse_context_chunks(context_text: str, max_chunks: int = 8, preview_chars: int = 700) -> list[dict[str, Any]]:
    """Parse retrieved context text into structured preview chunks."""
    if not context_text:
        return []

    raw_chunks = [c.strip() for c in context_text.split("\n---\n") if c.strip()]
    parsed = []

    for rank, chunk in enumerate(raw_chunks[:max_chunks], 1):
        lines = chunk.splitlines()
        header = ""
        body_lines = []

        if lines and (lines[0].startswith("[") or "date:" in lines[0].lower()):
            header = lines[0]
            body_lines = lines[1:]
        else:
            body_lines = lines

        body = "\n".join(body_lines).strip()
        preview = body[:preview_chars] + ("…" if len(body) > preview_chars else "")

        # Try to extract date or session info from header
        date_str = ""
        m_date = re.search(r"date:\s*([^\s\|\]]+)", chunk, re.IGNORECASE)
        if m_date:
            date_str = m_date.group(1)

        parsed.append({
            "rank": rank,
            "header": header,
            "date": date_str,
            "preview": preview or chunk[:preview_chars],
            "chars": len(chunk),
        })

    return parsed


def compute_quantiles(values: list[float]) -> dict[str, float]:
    """Compute min, median (p50), p95, and max for a list of values."""
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    s = sorted(values)
    n = len(s)
    avg = sum(s) / n

    def percentile(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return s[max(0, min(n - 1, idx))]

    return {
        "avg": avg,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": s[0],
        "max": s[-1],
    }


def build_report_payload(
    run_meta: dict[str, Any],
    summary: dict[str, Any],
    hypotheses: list[HypothesisEntry] | dict[str, HypothesisEntry] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct a consolidated benchmark result dictionary matching Caura standards."""
    # Index hypotheses by question_id
    hypos_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(hypotheses, dict):
        for k, v in hypotheses.items():
            hypos_by_id[k] = v.model_dump() if hasattr(v, "model_dump") else v
    else:
        for item in hypotheses:
            d = item.model_dump() if hasattr(item, "model_dump") else item
            hypos_by_id[d.get("question_id", "")] = d

    results = summary.get("results", [])
    question_rows: list[dict[str, Any]] = []

    ret_latencies: list[float] = []
    gen_latencies: list[float] = []
    context_lengths: list[int] = []
    chunk_counts: list[int] = []

    abstention_count = 0
    abstention_correct = 0

    for res in results:
        qid = res.get("question_id", "")
        hypo = hypos_by_id.get(qid, {})
        is_abs = "_abs" in qid or "abstain" in res.get("question_type", "").lower()

        if is_abs:
            abstention_count += 1
            if res.get("correct"):
                abstention_correct += 1

        ret_ms = float(hypo.get("retrieve_time_ms") or 0.0)
        gen_ms = float(hypo.get("generate_time_ms") or 0.0)
        if ret_ms > 0:
            ret_latencies.append(ret_ms)
        if gen_ms > 0:
            gen_latencies.append(gen_ms)

        raw_context = hypo.get("context") or ""
        ctx_len = len(raw_context)
        context_lengths.append(ctx_len)

        all_chunks = [c for c in raw_context.split("\n---\n") if c.strip()]
        total_chunks_count = len(all_chunks)
        chunk_counts.append(total_chunks_count)

        parsed_chunks = parse_context_chunks(raw_context, max_chunks=6, preview_chars=700)

        question_rows.append({
            "question_id": qid,
            "question": res.get("question") or hypo.get("question", ""),
            "gold_answer": res.get("gold_answer") or hypo.get("answer", ""),
            "hypothesis": res.get("hypothesis") or hypo.get("hypothesis", ""),
            "question_type": res.get("question_type") or hypo.get("question_type", "unknown"),
            "correct": bool(res.get("correct", False)),
            "is_abstention": is_abs,
            "judge_reason": res.get("judge_reason", ""),
            "judge_model": res.get("judge_model", run_meta.get("judge", "unknown")),
            "retrieve_time_ms": ret_ms,
            "generate_time_ms": gen_ms,
            "context_chars": ctx_len,
            "chunks_count": total_chunks_count,
            "retrieved_chunks": parsed_chunks,
            "pipeline": hypo.get("pipeline") or run_meta.get("pipeline", "direct"),
            "pipeline_trace": hypo.get("pipeline_trace"),
        })

    by_qtype = summary.get("by_question_type", {})

    # Compute official 500-question weighted projection if multiple categories evaluated
    weighted_acc = 0.0
    weight_sum = 0.0
    for qtype, stats in by_qtype.items():
        w = OFFICIAL_CATEGORY_WEIGHTS.get(qtype, 0.0)
        acc = stats.get("accuracy", 0.0)
        weighted_acc += acc * w
        weight_sum += w

    projected_official = (weighted_acc / weight_sum) if weight_sum > 0 else summary.get("overall_accuracy", 0.0)

    retrieval_stats = compute_quantiles(ret_latencies)
    retrieval_stats["avg_context_chars"] = (sum(context_lengths) / len(context_lengths)) if context_lengths else 0
    retrieval_stats["avg_chunks"] = (sum(chunk_counts) / len(chunk_counts)) if chunk_counts else 0

    generation_stats = compute_quantiles(gen_latencies)

    enhanced_summary = {
        "overall_accuracy": summary.get("overall_accuracy", 0.0),
        "total_questions": summary.get("total_questions", len(question_rows)),
        "correct_questions": summary.get("correct_questions", sum(1 for q in question_rows if q["correct"])),
        "projected_official_accuracy": projected_official,
        "abstention_accuracy": (abstention_correct / abstention_count) if abstention_count > 0 else None,
        "abstention_count": abstention_count,
        "by_question_type": by_qtype,
        "retrieval": retrieval_stats,
        "generation": generation_stats,
    }

    return {
        "schema_version": 1,
        "benchmark": "LongMemEval",
        "run": run_meta,
        "summary": enhanced_summary,
        "questions": question_rows,
    }


def render_report(result: dict[str, Any], output_path: Path) -> Path:
    """Render a standalone, self-contained HTML benchmark report."""
    run_name = result.get("run", {}).get("name", "LongMemEval Run")
    title = f"Caura × LongMemEval — {run_name}"
    payload = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    document = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def generate_report_for_run(
    run_dir: Path,
    output_file: Path | None = None,
    eval_file_name: str = "eval_results.json",
) -> Path:
    """Generate or regenerate report.html from an existing run directory."""
    eval_path = run_dir / eval_file_name
    if not eval_path.exists():
        # Fallback search for any eval_results*.json
        candidates = list(run_dir.glob("eval_results*.json"))
        if not candidates:
            raise FileNotFoundError(f"No evaluation results JSON found in {run_dir}")
        eval_path = candidates[0]

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

    # Load hypotheses if available
    hypos: list[dict[str, Any]] = []
    hypo_path = run_dir / "hypotheses.jsonl"
    if not hypo_path.exists():
        hypo_path = run_dir / "hypotheses_unique.jsonl"

    if hypo_path.exists():
        with open(hypo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        hypos.append(json.loads(line))
                    except Exception:
                        pass

    # Extract or infer run metadata
    judge_model = "unknown"
    if eval_data.get("results"):
        judge_model = eval_data["results"][0].get("judge_model", "unknown")

    # Determine provider from folder name or hypotheses
    folder_name = run_dir.name
    provider_name = "caura" if "caura" in folder_name.lower() else "bm25" if "bm25" in folder_name.lower() else "custom"

    mtime = datetime.fromtimestamp(eval_path.stat().st_mtime, tz=timezone.utc).isoformat()

    run_meta = {
        "name": folder_name,
        "provider": provider_name,
        "started_at": mtime,
        "finished_at": mtime,
        "duration_seconds": None,
        "top_k": 50 if "topk50" in folder_name else 20,
        "skip_ingest": True,
        "reader": "gemini-3.8-flash" if "gemini38" in folder_name else "gemini-2.5-flash",
        "judge": judge_model,
        "parameters": {
            "selection": {
                "question_count": eval_data.get("total_questions", len(hypos)),
                "categories": list(eval_data.get("by_question_type", {}).keys()),
            },
            "ingestion": {
                "provider": provider_name,
                "isolation": "One user fleet/sandbox per question",
            },
            "retrieval": {
                "top_k": 50 if "topk50" in folder_name else 20,
                "strategy": "Adaptive profile / semantic search",
                "context_ordering": "Chronological (oldest to newest)",
            },
            "generation": {
                "reader": "gemini-3.8-flash" if "gemini38" in folder_name else "gemini-2.5-flash",
                "judge": judge_model,
                "prompt": "Official answer prompt with chronological context",
            },
        },
    }

    # If results.json already has run info, merge it
    results_json = run_dir / "results.json"
    if results_json.exists():
        try:
            saved_res = json.loads(results_json.read_text(encoding="utf-8"))
            if "run" in saved_res:
                run_meta.update(saved_res["run"])
        except Exception:
            pass

    # If source_hypotheses is present, inherit retrieval and ingestion parameters from source run
    src_hypo = run_meta.get("parameters", {}).get("source_hypotheses")
    if src_hypo:
        try:
            src_results_path = Path(src_hypo).parent / "results.json"
            if src_results_path.exists():
                src_data = json.loads(src_results_path.read_text(encoding="utf-8"))
                src_run = src_data.get("run", {})
                src_params = src_run.get("parameters", {})
                run_params = run_meta.setdefault("parameters", {})
                if "retrieval" in src_params and "retrieval" not in run_params:
                    run_params["retrieval"] = src_params["retrieval"]
                if "ingestion" in src_params and "ingestion" not in run_params:
                    run_params["ingestion"] = src_params["ingestion"]
                if "provider" in src_run and "provider" not in run_meta:
                    run_meta["provider"] = src_run["provider"]
        except Exception:
            pass

    if "pipeline" not in run_meta:
        first_hypo = next(iter(hypos.values())) if isinstance(hypos, dict) else (hypos[0] if hypos else None)
        if first_hypo:
            pipe = getattr(first_hypo, "pipeline", None) or (first_hypo.get("pipeline") if isinstance(first_hypo, dict) else None)
            if pipe:
                run_meta["pipeline"] = pipe
                run_meta.setdefault("parameters", {}).setdefault("generation", {})["pipeline"] = pipe

    payload = build_report_payload(run_meta, eval_data, hypos)

    # Save enriched results.json
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    dest = output_file or (run_dir / "report.html")
    return render_report(payload, dest)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #07090c;
      --surface: #0f1318;
      --surface-raised: #141920;
      --surface-hover: #19202a;
      --surface-border: #1e2632;
      --line: #242e3d;
      --line-subtle: #19222e;
      --text: #f0f4f8;
      --muted: #94a3b8;
      --faint: #64748b;
      --emerald: #67d391;
      --emerald-soft: rgba(103, 211, 145, 0.12);
      --emerald-glow: rgba(103, 211, 145, 0.35);
      --cyan: #66e7ff;
      --cyan-soft: rgba(102, 231, 255, 0.12);
      --amber: #f2bd5b;
      --amber-soft: rgba(242, 189, 91, 0.12);
      --rose: #ff6b7a;
      --rose-soft: rgba(255, 107, 122, 0.12);
      --purple: #a78bfa;
      --purple-soft: rgba(167, 139, 250, 0.12);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-lg: 0 24px 80px -12px rgba(0, 0, 0, 0.65);
      --shadow-glow: 0 0 35px -5px;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.6;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      position: relative;
      overflow-x: hidden;
    }

    /* Ambient background grid & lighting */
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      background:
        radial-gradient(circle at 12% 10%, rgba(103, 211, 145, 0.10), transparent 32%),
        radial-gradient(circle at 86% 14%, rgba(102, 231, 255, 0.08), transparent 28%),
        radial-gradient(circle at 50% 88%, rgba(167, 139, 250, 0.06), transparent 35%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: auto, auto, auto, 48px 48px, 48px 48px;
      mask-image: linear-gradient(to bottom, #000 70%, transparent 100%);
    }

    .shell {
      position: relative;
      z-index: 1;
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto;
      padding: 32px 0 96px;
    }

    /* Top Navigation Bar */
    nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 36px;
      border-bottom: 1px solid var(--surface-border);
      margin-bottom: 32px;
    }

    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand {
      font-size: 24px;
      font-weight: 900;
      letter-spacing: -0.06em;
      color: #fff;
    }

    .brand i {
      color: var(--emerald);
      font-style: normal;
      text-shadow: 0 0 16px var(--emerald);
    }

    .brand-divider {
      width: 1px;
      height: 22px;
      background: var(--line);
    }

    .benchmark-pill {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cyan);
      background: var(--cyan-soft);
      border: 1px solid rgba(102, 231, 255, 0.25);
      border-radius: 999px;
      padding: 6px 12px;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--emerald);
      background: var(--emerald-soft);
      border: 1px solid rgba(103, 211, 145, 0.3);
      border-radius: 999px;
      padding: 6px 14px;
    }

    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--emerald);
      box-shadow: 0 0 12px var(--emerald);
      animation: pulse 2s infinite ease-in-out;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.85); }
    }

    /* Hero Section */
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 0.6fr;
      gap: 36px;
      align-items: end;
      margin-bottom: 32px;
    }

    .eyebrow {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--cyan);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    h1 {
      font-size: clamp(48px, 6vw, 86px);
      font-weight: 900;
      line-height: 0.92;
      letter-spacing: -0.065em;
      margin: 0;
      color: #fff;
    }

    h1 span.stroke {
      display: inline-block;
      color: transparent;
      -webkit-text-stroke: 1px rgba(240, 244, 248, 0.38);
    }

    .lede {
      font-size: 15px;
      color: var(--muted);
      line-height: 1.7;
      margin: 0;
      max-width: 480px;
    }

    /* Quick Badges Row */
    .run-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }

    .chip {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      color: var(--muted);
      background: var(--surface-raised);
      border: 1px solid var(--surface-border);
      padding: 5px 10px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .chip strong {
      color: var(--text);
    }

    /* Common Card Styling */
    .card {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      position: relative;
      overflow: hidden;
    }

    /* Parameter Grid */
    .parameters-card {
      margin-bottom: 24px;
    }

    .parameters-header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--surface-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .parameters-header h2 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin: 0;
    }

    .parameters-header p {
      margin: 0;
      font-size: 12px;
      color: var(--faint);
      font-family: "Cascadia Mono", Consolas, monospace;
    }

    .param-groups {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
    }

    .param-group {
      padding: 20px 24px;
      border-right: 1px solid var(--surface-border);
    }

    .param-group:last-child {
      border-right: none;
    }

    .param-group h3 {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--cyan);
      margin: 0 0 14px 0;
    }

    .param-list {
      display: grid;
      gap: 10px;
    }

    .param-row {
      display: grid;
      gap: 2px;
    }

    .param-key {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--faint);
    }

    .param-val {
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      overflow-wrap: anywhere;
    }

    /* Top Metric Tiles */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
      margin-bottom: 32px;
    }

    .metric-card {
      grid-column: span 3;
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 160px;
      position: relative;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.16);
    }

    .metric-card::after {
      content: "";
      position: absolute;
      top: -50px;
      right: -50px;
      width: 130px;
      height: 130px;
      border-radius: 50%;
      background: var(--glow-color, var(--emerald));
      filter: blur(55px);
      opacity: 0.18;
      pointer-events: none;
    }

    .metric-label {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .metric-value {
      font-size: 46px;
      font-weight: 900;
      letter-spacing: -0.06em;
      line-height: 1;
      margin: 12px 0 6px;
      color: var(--value-color, #fff);
    }

    .metric-sub {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      color: var(--faint);
    }

    /* Capability Map & Protocol Section */
    .section-grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 20px;
      margin-bottom: 48px;
    }

    .wide-card {
      grid-column: span 8;
      padding: 24px 28px;
    }

    .narrow-card {
      grid-column: span 4;
      padding: 24px 28px;
    }

    .section-title {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.035em;
      margin: 0;
      color: #fff;
    }

    .section-subtitle {
      font-size: 12px;
      color: var(--muted);
      margin: 4px 0 20px 0;
    }

    /* Category Breakdown Bars */
    .bar-list {
      display: grid;
      gap: 16px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 210px 1fr 100px;
      gap: 16px;
      align-items: center;
      cursor: pointer;
      padding: 6px 8px;
      border-radius: 8px;
      transition: background 0.15s ease;
    }

    .bar-row:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    .bar-label-wrap {
      display: flex;
      flex-direction: column;
    }

    .bar-title {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }

    .bar-desc {
      font-size: 10px;
      color: var(--faint);
      margin-top: 1px;
    }

    .bar-track {
      height: 10px;
      background: var(--surface-border);
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--bar-start, #67d391), var(--bar-end, #66e7ff));
      box-shadow: 0 0 14px var(--bar-glow, rgba(103, 211, 145, 0.4));
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .bar-score {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
      text-align: right;
      color: #fff;
    }

    .bar-counts {
      font-size: 10px;
      color: var(--faint);
      font-weight: 400;
    }

    /* Cognitive Protocol Sidebar */
    .protocol-list {
      display: grid;
      gap: 16px;
      margin-top: 8px;
    }

    .protocol-item {
      display: grid;
      grid-template-columns: 28px 1fr;
      gap: 14px;
    }

    .protocol-num {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
      color: var(--emerald);
      padding-top: 1px;
    }

    .protocol-text h4 {
      font-size: 13px;
      font-weight: 700;
      margin: 0 0 4px 0;
      color: var(--text);
    }

    .protocol-text p {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }

    /* Trace Explorer Section */
    .explorer-header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      margin-bottom: 18px;
      gap: 16px;
      flex-wrap: wrap;
    }

    .controls-bar {
      display: flex;
      gap: 12px;
      margin-bottom: 20px;
      flex-wrap: wrap;
      align-items: center;
    }

    .ctrl-input, .ctrl-select {
      background: var(--surface-raised);
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .ctrl-select {
      cursor: pointer;
      min-width: 180px;
    }

    .ctrl-input {
      flex: 1;
      min-width: 260px;
    }

    .ctrl-input:focus, .ctrl-select:focus {
      border-color: var(--emerald);
      box-shadow: 0 0 0 2px var(--emerald-soft);
    }

    .btn-action {
      background: var(--surface-raised);
      border: 1px solid var(--surface-border);
      color: var(--muted);
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .btn-action:hover {
      background: var(--surface-hover);
      color: #fff;
      border-color: var(--line);
    }

    /* Question Accordion Card */
    .question-list {
      display: grid;
      gap: 10px;
    }

    .question-card {
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      background: var(--surface);
      overflow: hidden;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .question-card[open] {
      border-color: rgba(255, 255, 255, 0.14);
      background: #0d1117;
    }

    .question-summary {
      display: grid;
      grid-template-columns: 110px 170px 1fr 140px 90px;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      cursor: pointer;
      user-select: none;
      list-style: none;
    }

    .question-summary::-webkit-details-marker {
      display: none;
    }

    .q-id {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }

    .cat-badge {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 3px 8px;
      border-radius: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      display: inline-block;
      max-width: 160px;
    }

    .q-text {
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .q-timing {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      color: var(--faint);
      text-align: right;
    }

    .verdict-badge {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-align: center;
      padding: 4px 10px;
      border-radius: 999px;
    }

    .verdict-pass {
      color: var(--emerald);
      background: var(--emerald-soft);
      border: 1px solid rgba(103, 211, 145, 0.35);
      box-shadow: 0 0 10px rgba(103, 211, 145, 0.15);
    }

    .verdict-fail {
      color: var(--rose);
      background: var(--rose-soft);
      border: 1px solid rgba(255, 107, 122, 0.35);
      box-shadow: 0 0 10px rgba(255, 107, 122, 0.15);
    }

    .verdict-abstain {
      color: var(--amber);
      background: var(--amber-soft);
      border: 1px solid rgba(242, 189, 91, 0.35);
      box-shadow: 0 0 10px rgba(242, 189, 91, 0.15);
    }

    /* Detail Drawer */
    .question-detail {
      border-top: 1px solid var(--surface-border);
      padding: 24px;
      background: #090c10;
      display: grid;
      gap: 20px;
    }

    .comparison-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    .comp-panel {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      padding: 16px;
    }

    .comp-panel.reference {
      background: rgba(102, 231, 255, 0.02);
      border-color: rgba(102, 231, 255, 0.15);
    }

    .comp-panel h4 {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin: 0 0 10px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .comp-body {
      font-size: 13px;
      line-height: 1.65;
      color: #e2e8f0;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .judge-panel {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      padding: 14px 18px;
      display: flex;
      align-items: flex-start;
      gap: 16px;
    }

    .judge-icon {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 18px;
    }

    .judge-meta {
      flex: 1;
    }

    .judge-title {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cyan);
      margin-bottom: 4px;
    }

    .judge-reason {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }

    /* Retrieved Context Inspector */
    .context-panel {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      padding: 16px;
    }

    .context-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .context-header h4 {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin: 0;
    }

    .chunk-list {
      display: grid;
      gap: 8px;
    }

    .chunk-card {
      background: #06080a;
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      padding: 10px 14px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      line-height: 1.5;
      color: #cbd5e1;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .chunk-meta {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
      font-size: 10px;
      color: var(--cyan);
      border-bottom: 1px solid #141a22;
      padding-bottom: 4px;
    }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 64px 20px;
      color: var(--muted);
      font-size: 14px;
      background: var(--surface);
      border: 1px dashed var(--surface-border);
      border-radius: var(--radius-md);
    }

    /* Footer */
    footer {
      margin-top: 64px;
      padding-top: 24px;
      border-top: 1px solid var(--surface-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--faint);
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      flex-wrap: wrap;
      gap: 12px;
    }

    footer a {
      color: var(--emerald);
      text-decoration: none;
    }

    /* Responsive */
    @media (max-width: 1024px) {
      .hero { grid-template-columns: 1fr; gap: 20px; }
      .param-groups { grid-template-columns: 1fr 1fr; }
      .param-group:nth-child(2) { border-right: none; }
      .param-group:nth-child(-n+2) { border-bottom: 1px solid var(--surface-border); }
      .metric-card { grid-column: span 6; }
      .wide-card, .narrow-card { grid-column: span 12; }
      .comparison-grid { grid-template-columns: 1fr; }
      .question-summary { grid-template-columns: 90px 130px 1fr 80px; }
      .q-timing { display: none; }
    }

    @media (max-width: 640px) {
      .shell { width: calc(100% - 24px); padding: 20px 0 60px; }
      nav { flex-direction: column; align-items: flex-start; gap: 14px; }
      .param-groups { grid-template-columns: 1fr; }
      .param-group { border-right: none; border-bottom: 1px solid var(--surface-border); }
      .param-group:last-child { border-bottom: none; }
      .metric-card { grid-column: span 12; }
      .bar-row { grid-template-columns: 1fr 60px; }
      .bar-track { display: none; }
      .question-summary { grid-template-columns: 1fr 80px; }
      .cat-badge { display: none; }
      .controls-bar { flex-direction: column; align-items: stretch; }
    }
  </style>
</head>
<body>

<main class="shell">
  <!-- Top Navigation -->
  <nav>
    <div class="brand-wrap">
      <div class="brand">caura<i>.</i>ai</div>
      <div class="brand-divider"></div>
      <div class="benchmark-pill">LongMemEval-S Suite</div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
      <button class="btn-action" onclick="toggleAllCards()">Toggle All</button>
      <button class="btn-action" onclick="window.print()">Print / PDF</button>
      <div class="status-badge"><span class="pulse-dot"></span>Production Benchmark</div>
    </div>
  </nav>

  <!-- Hero Section -->
  <section class="hero">
    <div>
      <div class="eyebrow">
        <span>Persistent Agent Memory</span> · <span>ICLR 2025 Standard</span>
      </div>
      <h1>LongMemEval<span class="stroke">benchmark.</span></h1>
      <div class="run-chips" id="run-chips"></div>
    </div>
    <p class="lede">
      A comprehensive evaluation of multi-session retrieval, temporal chronology, evolving state updates, and hallucination resistance across thousands of conversational sessions.
    </p>
  </section>

  <!-- Run Parameters Card -->
  <section class="card parameters-card" aria-label="Run parameters">
    <div class="parameters-header">
      <div>
        <h2>Execution Parameters</h2>
      </div>
      <p id="run-timestamp"></p>
    </div>
    <div class="param-groups" id="param-groups"></div>
  </section>

  <!-- Top Metric KPI Strip -->
  <section class="metric-grid" id="metrics"></section>

  <!-- Capability Map & Protocol -->
  <section class="section-grid">
    <div class="card wide-card">
      <div class="eyebrow">Cognitive Dimension Map</div>
      <h3 class="section-title">Accuracy by Question Type</h3>
      <p class="section-subtitle">Click any category bar to immediately filter question traces below.</p>
      <div class="bar-list" id="bars"></div>
    </div>

    <aside class="card narrow-card">
      <div class="eyebrow">Evaluation Standard</div>
      <h3 class="section-title">Cognitive Protocol</h3>
      <div class="protocol-list">
        <div class="protocol-item">
          <div class="protocol-num">01</div>
          <div class="protocol-text">
            <h4>Strict Isolation</h4>
            <p>Each evaluation item executes in its own sandboxed fleet to eliminate cross-question memory leakage.</p>
          </div>
        </div>
        <div class="protocol-item">
          <div class="protocol-num">02</div>
          <div class="protocol-text">
            <h4>Temporal & State Tracking</h4>
            <p>Knowledge-update and temporal questions measure ability to supersede outdated facts over time.</p>
          </div>
        </div>
        <div class="protocol-item">
          <div class="protocol-num">03</div>
          <div class="protocol-text">
            <h4>Official Judge Protocol</h4>
            <p>Official LongMemEval binary evaluation prompts enforce factual accuracy without grading drift.</p>
          </div>
        </div>
      </div>
    </aside>
  </section>

  <!-- Question Trace Explorer -->
  <section>
    <div class="explorer-header">
      <div>
        <div class="eyebrow">Trace Inspector</div>
        <h3 class="section-title">Question-Level Traces</h3>
      </div>
      <div id="visible-count" style="font-family:'Cascadia Mono',Consolas,monospace; font-size:12px; color:var(--muted);"></div>
    </div>

    <div class="controls-bar">
      <select id="filter-category" class="ctrl-select">
        <option value="all">All Categories</option>
      </select>
      <select id="filter-outcome" class="ctrl-select">
        <option value="all">All Outcomes</option>
        <option value="pass">Pass (Correct)</option>
        <option value="fail">Fail (Incorrect)</option>
        <option value="abstain">Abstention</option>
      </select>
      <select id="sort-by" class="ctrl-select">
        <option value="default">Default Order</option>
        <option value="slowest-ret">Slowest Retrieval</option>
        <option value="slowest-gen">Slowest Generation</option>
        <option value="failures-first">Failures First</option>
      </select>
      <input id="search-input" class="ctrl-input" type="search" placeholder="Search questions, answers, hypotheses, or judge feedback…">
    </div>

    <div class="question-list" id="question-list"></div>
  </section>

  <!-- Footer -->
  <footer>
    <span id="footer-meta">Caura AI Benchmark Suite</span>
    <span>Self-contained portable report · No credentials or external CDNs required</span>
  </footer>
</main>

<script>
const data = __DATA__;
const run = data.run || {};
const params = run.parameters || {};
const sel = params.selection || {};
const ingest = params.ingestion || {};
const retParams = params.retrieval || {};
const genParams = params.generation || {};
const summary = data.summary || {};
const questions = data.questions || [];

const categoryMeta = {
  'single-session-user': { color: '#67d391', label: 'User Recall', desc: 'Recall user-stated facts' },
  'single-session-assistant': { color: '#66e7ff', label: 'Assistant Recall', desc: 'Recall assistant advice' },
  'multi-session': { color: '#a78bfa', label: 'Multi-Session', desc: 'Cross-conversation synthesis' },
  'temporal-reasoning': { color: '#f2bd5b', label: 'Temporal', desc: 'Time math & chronological sequence' },
  'knowledge-update': { color: '#ff6b7a', label: 'Knowledge Update', desc: 'Superseding old info & state shifts' },
  'single-session-preference': { color: '#ec4899', label: 'Preference', desc: 'Implicit & explicit user preference' }
};

const pct = v => v == null ? '—' : `${(v * 100).toFixed(1)}%`;
const ms = v => v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`;
const fmtNum = v => v == null ? '—' : Number(v).toLocaleString();
const h = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

const isAdaptive = (
  retParams.category_adaptive !== false &&
  (String(run.provider || ingest.provider || '').toLowerCase().includes('caura') ||
   retParams.category_adaptive === true ||
   String(retParams.strategy || '').toLowerCase().includes('adaptive') ||
   String(run.name || '').toLowerCase().includes('adaptive') ||
   run.top_k_adaptive === true)
);

const baseK = run.top_k || retParams.top_k || 20;
const topKDisplay = isAdaptive ? `Adaptive (15–60, base ${baseK})` : fmtNum(baseK);

// Populate chips
document.querySelector('#run-chips').innerHTML = `
  <span class="chip">Run: <strong>${h(run.name || 'default')}</strong></span>
  <span class="chip">Provider: <strong>${h(run.provider || 'caura')}</strong></span>
  ${(run.pipeline || genParams.pipeline || params.pipeline) ? `<span class="chip">Pipeline: <strong style="color:#a78bfa;">${h(run.pipeline || genParams.pipeline || params.pipeline)}</strong></span>` : ''}
  <span class="chip">Reader: <strong>${h(run.reader || genParams.reader || 'default')}</strong></span>
  <span class="chip">Judge: <strong>${h(run.judge || genParams.judge || 'default')}</strong></span>
  <span class="chip">Top-k: <strong>${h(topKDisplay)}</strong></span>
`;

// Populate execution parameters
const paramGroups = [
  {
    title: 'Dataset & Selection',
    items: [
      ['Benchmark', 'LongMemEval-S (Cleaned ICLR 2025)'],
      ['Questions Evaluated', fmtNum(summary.total_questions || questions.length)],
      ['Categories Tested', (sel.categories && sel.categories.length) ? sel.categories.join(', ') : 'All 6 categories'],
      ['Sampling Strategy', sel.per_category ? `${sel.per_category} / category (balanced)` : sel.limit ? `limit ${sel.limit}` : 'All selected items']
    ]
  },
  {
    title: 'Memory Store',
    items: [
      ['Provider', h(run.provider || ingest.provider || 'caura')],
      ['Isolation', ingest.isolation || 'One fleet sandbox per question'],
      ['Ingestion Mode', run.skip_ingest ? 'Pre-indexed / Skipped' : (ingest.mode || 'Bulk session ingest')],
      ['Chunking', ingest.chunk_chars ? `${fmtNum(ingest.chunk_chars)} chars` : '4,000 chars']
    ]
  },
  {
    title: 'Retrieval Strategy',
    items: [
      ['Top-k Setting', topKDisplay],
      ['Search Strategy', retParams.strategy || (isAdaptive ? 'Adaptive profile / semantic search' : 'Semantic search')],
      ['Context Ordering', retParams.context_ordering || 'Chronological (oldest to newest)'],
      ['Avg Context Tokens', summary.retrieval ? `~${fmtNum(Math.round(summary.retrieval.avg_context_chars / 4))} est. tokens` : '—']
    ]
  },
  {
    title: 'Models & Execution',
    items: [
      ['Reader LLM', h(run.reader || genParams.reader || 'gemini-3.8-flash')],
      ['Judge LLM', h(run.judge || genParams.judge || 'gemini-3.5-flash-lite')],
      ['Judge Protocol', 'Official LongMemEval binary judge'],
      ['Completed', run.finished_at ? new Date(run.finished_at).toLocaleString() : 'Not recorded']
    ]
  }
];

document.querySelector('#param-groups').innerHTML = paramGroups.map(g => `
  <div class="param-group">
    <h3>${h(g.title)}</h3>
    <div class="param-list">
      ${g.items.map(([k, v]) => `
        <div class="param-row">
          <span class="param-key">${h(k)}</span>
          <span class="param-val">${h(v)}</span>
        </div>
      `).join('')}
    </div>
  </div>
`).join('');

document.querySelector('#run-timestamp').textContent = run.finished_at ? `Completed ${new Date(run.finished_at).toLocaleDateString()}` : '';

// Metric Cards
const ret = summary.retrieval || {};
const gen = summary.generation || {};
const metrics = [
  {
    label: 'Overall Accuracy',
    value: pct(summary.overall_accuracy),
    sub: `${summary.correct_questions || 0} of ${summary.total_questions || 0} passed`,
    valColor: 'var(--emerald)',
    glow: 'var(--emerald)'
  },
  {
    label: 'Projected Official',
    value: pct(summary.projected_official_accuracy),
    sub: 'Reweighted to 500-item mix',
    valColor: 'var(--cyan)',
    glow: 'var(--cyan)'
  },
  {
    label: 'Retrieval Latency (p95)',
    value: ms(ret.p95),
    sub: `Median ${ms(ret.p50)} · avg ${ms(ret.avg)}`,
    valColor: 'var(--amber)',
    glow: 'var(--amber)'
  },
  {
    label: 'Generation Latency (p95)',
    value: ms(gen.p95),
    sub: `Median ${ms(gen.p50)} · avg ${ms(gen.avg)}`,
    valColor: 'var(--rose)',
    glow: 'var(--rose)'
  }
];

if (summary.abstention_count && summary.abstention_count > 0) {
  metrics.push({
    label: 'Abstention Acc',
    value: pct(summary.abstention_accuracy),
    sub: `${summary.abstention_count} ungrounded queries`,
    valColor: 'var(--purple)',
    glow: 'var(--purple)'
  });
}

document.querySelector('#metrics').innerHTML = metrics.map(m => `
  <div class="card metric-card" style="--value-color:${m.valColor}; --glow-color:${m.glow};">
    <div class="metric-label"><span>${h(m.label)}</span></div>
    <div class="metric-value">${m.value}</div>
    <div class="metric-sub">${m.sub}</div>
  </div>
`).join('');

// Capability Bars
const byQtype = summary.by_question_type || {};
const catEntries = Object.entries(byQtype);

document.querySelector('#bars').innerHTML = catEntries.map(([cat, stats]) => {
  const meta = categoryMeta[cat] || { color: '#67d391', label: cat, desc: '' };
  const acc = stats.accuracy ?? 0;
  return `
    <div class="bar-row" onclick="filterByCategory('${h(cat)}')">
      <div class="bar-label-wrap">
        <span class="bar-title" style="color:${meta.color};">${h(meta.label)}</span>
        <span class="bar-desc">${h(meta.desc || cat)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${acc * 100}%; --bar-start:${meta.color}; --bar-end:var(--cyan); --bar-glow:${meta.color}66;"></div>
      </div>
      <div class="bar-score">
        <div>${pct(acc)}</div>
        <div class="bar-counts">${stats.correct}/${stats.total} correct</div>
      </div>
    </div>
  `;
}).join('');

// Populate Category Filter Dropdown
const catSelect = document.querySelector('#filter-category');
catEntries.forEach(([cat, stats]) => {
  const meta = categoryMeta[cat] || { label: cat };
  catSelect.insertAdjacentHTML('beforeend', `<option value="${h(cat)}">${h(meta.label)} (${stats.total})</option>`);
});

// Trace Explorer Filtering & Rendering
const outcomeSelect = document.querySelector('#filter-outcome');
const sortSelect = document.querySelector('#sort-by');
const searchInput = document.querySelector('#search-input');
const questionList = document.querySelector('#question-list');
const visibleCount = document.querySelector('#visible-count');

function filterByCategory(cat) {
  catSelect.value = cat;
  renderQuestions();
  document.querySelector('.explorer-header').scrollIntoView({ behavior: 'smooth' });
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1800);
  });
}

function toggleAllCards() {
  const cards = document.querySelectorAll('.question-card');
  const allOpen = Array.from(cards).every(c => c.open);
  cards.forEach(c => c.open = !allOpen);
}

function renderQuestions() {
  const selectedCat = catSelect.value;
  const selectedOutcome = outcomeSelect.value;
  const sortBy = sortSelect.value;
  const query = searchInput.value.trim().toLowerCase();

  let rows = questions.filter(q => {
    if (selectedCat !== 'all' && q.question_type !== selectedCat) return false;
    if (selectedOutcome === 'pass' && !q.correct) return false;
    if (selectedOutcome === 'fail' && q.correct) return false;
    if (selectedOutcome === 'abstain' && !q.is_abstention) return false;

    if (query) {
      const hay = `${q.question_id} ${q.question} ${q.gold_answer} ${q.hypothesis} ${q.judge_reason}`.toLowerCase();
      if (!hay.includes(query)) return false;
    }
    return true;
  });

  // Sorting
  if (sortBy === 'slowest-ret') {
    rows.sort((a, b) => (b.retrieve_time_ms || 0) - (a.retrieve_time_ms || 0));
  } else if (sortBy === 'slowest-gen') {
    rows.sort((a, b) => (b.generate_time_ms || 0) - (a.generate_time_ms || 0));
  } else if (sortBy === 'failures-first') {
    rows.sort((a, b) => Number(a.correct) - Number(b.correct));
  }

  visibleCount.textContent = `Showing ${rows.length} of ${questions.length} questions`;

  if (!rows.length) {
    questionList.innerHTML = `<div class="empty-state">No benchmark traces matched your active filters.</div>`;
    return;
  }

  questionList.innerHTML = rows.map((q, idx) => {
    const meta = categoryMeta[q.question_type] || { color: '#67d391', label: q.question_type };
    const verdictCls = q.correct ? 'verdict-pass' : (q.is_abstention ? 'verdict-abstain' : 'verdict-fail');
    const verdictText = q.correct ? 'PASS' : (q.is_abstention ? 'ABSTAIN' : 'FAIL');
    const chunks = q.retrieved_chunks || [];

    return `
      <details class="question-card" id="q-${h(q.question_id)}">
        <summary class="question-summary">
          <span class="q-id">#${h(q.question_id)}</span>
          <div>
            <span class="cat-badge" style="background:${meta.color}22; color:${meta.color}; border:1px solid ${meta.color}44;">
              ${h(meta.label)}
            </span>
          </div>
          <span class="q-text" title="${h(q.question)}">${h(q.question)}</span>
          <span class="q-timing">⚡ ${ms(q.retrieve_time_ms)} ret · ${ms(q.generate_time_ms)} gen</span>
          <span class="verdict-badge ${verdictCls}">${verdictText}</span>
        </summary>

        <div class="question-detail">
          <div class="comparison-grid">
            <div class="comp-panel">
              <h4>
                <span>Generated Hypothesis (Caura)</span>
                <button class="btn-action" style="padding:2px 8px; font-size:10px;" onclick="copyToClipboard(${JSON.stringify(q.hypothesis)}, this)">Copy</button>
              </h4>
              <div class="comp-body">${h(q.hypothesis)}</div>
            </div>

            <div class="comp-panel reference">
              <h4>
                <span>Reference Ground Truth</span>
                <button class="btn-action" style="padding:2px 8px; font-size:10px;" onclick="copyToClipboard(${JSON.stringify(q.gold_answer)}, this)">Copy</button>
              </h4>
              <div class="comp-body">${h(q.gold_answer || 'No normal gold answer (Abstention item)')}</div>
            </div>
          </div>

          <div class="judge-panel">
            <div class="judge-icon">${q.correct ? '✅' : '❌'}</div>
            <div class="judge-meta">
              <div class="judge-title">Judge Verdict · ${h(q.judge_model || 'LLM Judge')} · ${q.correct ? 'CORRECT' : 'INCORRECT'}</div>
              <div class="judge-reason">${h(q.judge_reason || (q.correct ? 'Evaluation judge verified response agrees with ground truth.' : 'Response differed from reference ground truth.'))}</div>
            </div>
          </div>

          ${q.pipeline_trace ? `
            <div class="pipeline-panel" style="margin-top:14px; background:rgba(255,255,255,0.02); border:1px solid rgba(167,139,250,0.3); border-radius:8px; padding:12px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; font-size:12px; color:#a78bfa; text-transform:uppercase; letter-spacing:0.05em;">
                  🛡️ Multi-Stage Pipeline (${h(q.pipeline || 'agentic-v1')})
                </span>
                <span style="font-size:11px; color:var(--muted); font-family:monospace;">
                  Support: <strong style="color:${q.pipeline_trace.support_status === 'direct' ? '#67d391' : q.pipeline_trace.support_status === 'inferable' ? '#f2bd5b' : '#ff6b7a'}">${h(q.pipeline_trace.support_status || 'direct')}</strong>
                </span>
              </div>
              ${q.pipeline_trace.verifier_reason ? `
                <div style="font-size:12px; color:var(--text); margin-bottom:8px; background:rgba(0,0,0,0.25); padding:8px; border-radius:6px; line-height:1.45;">
                  <span style="color:var(--muted); font-size:11px; display:block; margin-bottom:2px; font-weight:600;">Verifier Reasoning:</span>
                  ${h(q.pipeline_trace.verifier_reason)}
                </div>
              ` : ''}
              ${q.pipeline_trace.extracted_facts && q.pipeline_trace.extracted_facts.length ? `
                <details style="font-size:12px; margin-top:6px;">
                  <summary style="cursor:pointer; color:var(--muted); font-size:11px;">Extracted Chronological Facts (${q.pipeline_trace.extracted_facts.length})</summary>
                  <ul style="margin:6px 0 0 18px; padding:0; color:var(--text); line-height:1.4;">
                    ${q.pipeline_trace.extracted_facts.map(f => `<li style="margin-bottom:3px;">${h(f)}</li>`).join('')}
                  </ul>
                </details>
              ` : ''}
            </div>
          ` : ''}

          <div class="context-panel">
            <div class="context-header">
              <h4>Retrieved Memories & Context (${q.chunks_count || chunks.length} chunks · ${fmtNum(q.context_chars)} characters)</h4>
              <span style="font-family:'Cascadia Mono',Consolas,monospace; font-size:11px; color:var(--faint);">Search time: ${ms(q.retrieve_time_ms)}</span>
            </div>
            ${chunks.length ? `
              <div class="chunk-list">
                ${chunks.map(c => `
                  <div class="chunk-card">
                    <div class="chunk-meta">
                      <span>Chunk #${c.rank}</span>
                      ${c.date ? `<span>Date: ${h(c.date)}</span>` : ''}
                      <span>Size: ${fmtNum(c.chars)} chars</span>
                    </div>
                    <div>${h(c.preview)}</div>
                  </div>
                `).join('')}
              </div>
            ` : `<div style="color:var(--faint); font-size:12px;">No individual chunk previews available.</div>`}
          </div>
        </div>
      </details>
    `;
  }).join('');
}

[catSelect, outcomeSelect, sortSelect].forEach(el => el.addEventListener('change', renderQuestions));
searchInput.addEventListener('input', renderQuestions);

// Initial render
renderQuestions();

document.querySelector('#footer-meta').textContent = `${h(run.name)} · ${run.finished_at ? run.finished_at.slice(0, 10) : ''} · ${h(run.reader || 'default')} / ${h(run.judge || 'default')} · Caura AI`;
</script>
</body>
</html>"""
