"""Per-question token and latency budget for a LongMemEval run.

Reports, for the contexts saved in ``hypotheses.jsonl``:

- context tokens per question (median / p95 / max), counted with the reader
  model's own tokenizer via the Gemini ``count_tokens`` API (not chars/4);
- total reader tokens per question summed over every reader call (extract reads
  the full context, then answer / infer / verify, retries and fallbacks
  included). Exact when the run recorded provider usage (``reader_usage``);
  otherwise reconstructed from the saved pipeline trace and marked as an
  estimate (lower bound: retries are not visible in the trace);
- retrieval and generation latency medians.

Counts are cached in ``<run>/context_tokens.json`` so re-running is free.

Usage:
    uv run python scripts/context_tokens.py outputs/<run> [--model gemini-3.8-flash] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from longmemeval.models import EvidenceBundle  # noqa: E402
from longmemeval.prompts import (  # noqa: E402
    build_evidence_answer_prompt,
    build_extract_evidence_prompt,
    build_infer_answer_prompt,
    build_verify_answer_prompt,
)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


class Counter:
    def __init__(self, model: str):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required to count tokens with the reader's tokenizer")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def count(self, text: str) -> int:
        for attempt in range(5):
            try:
                return int(self.client.models.count_tokens(model=self.model, contents=text).total_tokens)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        return 0


def reconstruct_prompts(h: dict, question_date: str | None) -> list[tuple[str, str]]:
    """Rebuild the agentic-v1 prompts this question must have seen (no retries)."""
    trace = h.get("pipeline_trace") or {}
    q = h.get("question") or ""
    ctx = h.get("context") or ""
    prompts = [("extract", build_extract_evidence_prompt(q, ctx, question_date))]
    if h.get("pipeline") != "agentic-v1":
        return [("direct", build_extract_evidence_prompt(q, ctx, question_date))]
    bundle = EvidenceBundle(
        str(trace.get("evidence_status") or "unsupported"),
        tuple(trace.get("facts") or ()),
        tuple(trace.get("requirements") or ()),
    )
    ev = bundle.as_context()
    prompts.append(("answer", build_evidence_answer_prompt(q, ev, question_date)))
    inference = trace.get("inference_answer")
    if inference:
        if bundle.facts:
            prompts.append(("infer", build_infer_answer_prompt(q, ev, question_date)))
        else:
            # Direct fallback re-reads the full context.
            prompts.append(("direct_fallback", build_extract_evidence_prompt(q, ctx, question_date)))
    candidates = (trace.get("initial_answer") or "",) + ((inference,) if inference else ())
    prompts.append(("verify", build_verify_answer_prompt(q, ev, candidates, question_date)))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=os.environ.get("READER_MODEL", "gemini-3.8-flash"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-path", type=Path, default=None)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists() and (ROOT / "outputs" / run_dir).exists():
        run_dir = ROOT / "outputs" / run_dir

    hyps: dict[str, dict] = {}
    with open(run_dir / "hypotheses.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                hyps[d["question_id"]] = d

    from longmemeval.dataset import LongMemEvalDataset

    dates = {it.question_id: it.question_date for it in LongMemEvalDataset(data_path=args.data_path).load_items()}

    cache_path = run_dir / "context_tokens.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8")).get("questions", {})

    counter = Counter(args.model)
    exact = all(h.get("reader_usage") for h in hyps.values())

    def work(qid: str) -> tuple[str, dict]:
        h = hyps[qid]
        row = cache.get(qid) or {}
        if row.get("model") == args.model and "context_tokens" in row and ("reader_total_tokens" in row):
            return qid, row
        ctx = h.get("context") or ""
        row = {
            "model": args.model,
            "context_chars": len(ctx),
            "context_tokens": counter.count(ctx) if ctx else 0,
            "retrieve_ms": h.get("retrieve_time_ms", 0.0),
            "generate_ms": h.get("generate_time_ms", 0.0),
        }
        usage = h.get("reader_usage")
        if usage:
            row["reader_calls"] = usage.get("n_calls")
            row["reader_prompt_tokens"] = usage.get("prompt_tokens")
            row["reader_completion_tokens"] = usage.get("completion_tokens")
            row["reader_reasoning_tokens"] = usage.get("reasoning_tokens")
            row["reader_total_tokens"] = usage.get("total_tokens")
            row["reader_tokens_exact"] = True
        else:
            prompts = reconstruct_prompts(h, dates.get(qid))
            prompt_tokens = sum(counter.count(p) for _, p in prompts)
            row["reader_calls"] = len(prompts)
            row["reader_prompt_tokens"] = prompt_tokens
            row["reader_completion_tokens"] = None
            row["reader_total_tokens"] = prompt_tokens
            row["reader_tokens_exact"] = False
        return qid, row

    qids = list(hyps)
    rows: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for qid, row in pool.map(work, qids):
            rows[qid] = row
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(qids)}", file=sys.stderr)
                cache_path.write_text(json.dumps({"model": args.model, "questions": rows}, indent=0), encoding="utf-8")
    cache_path.write_text(json.dumps({"model": args.model, "questions": rows}, indent=0), encoding="utf-8")

    ctx_tok = [r["context_tokens"] for r in rows.values()]
    ctx_chars = [r["context_chars"] for r in rows.values()]
    total_tok = [r["reader_total_tokens"] for r in rows.values() if r.get("reader_total_tokens") is not None]
    prompt_tok = [r["reader_prompt_tokens"] for r in rows.values() if r.get("reader_prompt_tokens") is not None]
    comp_tok = [r["reader_completion_tokens"] for r in rows.values() if r.get("reader_completion_tokens") is not None]
    calls = [r["reader_calls"] for r in rows.values() if r.get("reader_calls") is not None]
    ret_ms = [r["retrieve_ms"] for r in rows.values()]
    gen_ms = [r["generate_ms"] for r in rows.values()]

    label = "exact (provider usage)" if exact else "reconstructed from trace (lower bound: retries not counted)"
    print(f"\nRun: {run_dir}   reader tokenizer: {args.model}   questions: {len(rows)}")
    print(f"Reader token accounting: {label}\n")
    print("| metric | median | p95 | max |")
    print("|---|---|---|---|")
    print(f"| context characters | {statistics.median(ctx_chars):,.0f} | {pct(ctx_chars, .95):,.0f} | {max(ctx_chars):,} |")
    print(f"| context tokens ({args.model} tokenizer) | {statistics.median(ctx_tok):,.0f} | {pct(ctx_tok, .95):,.0f} | {max(ctx_tok):,} |")
    print(f"| chars/4 estimate for comparison | {statistics.median(ctx_chars)/4:,.0f} | {pct(ctx_chars, .95)/4:,.0f} | {max(ctx_chars)/4:,.0f} |")
    if calls:
        print(f"| reader calls per question | {statistics.median(calls):.0f} | {pct(calls, .95):.0f} | {max(calls)} |")
    if prompt_tok:
        print(f"| reader prompt tokens, all calls | {statistics.median(prompt_tok):,.0f} | {pct(prompt_tok, .95):,.0f} | {max(prompt_tok):,} |")
    if comp_tok:
        print(f"| reader completion tokens, all calls | {statistics.median(comp_tok):,.0f} | {pct(comp_tok, .95):,.0f} | {max(comp_tok):,} |")
    if total_tok:
        print(f"| reader total tokens, all calls | {statistics.median(total_tok):,.0f} | {pct(total_tok, .95):,.0f} | {max(total_tok):,} |")
    print(f"| retrieval latency (s) | {statistics.median(ret_ms)/1000:.1f} | {pct(ret_ms, .95)/1000:.1f} | {max(ret_ms)/1000:.1f} |")
    print(f"| generation latency (s) | {statistics.median(gen_ms)/1000:.1f} | {pct(gen_ms, .95)/1000:.1f} | {max(gen_ms)/1000:.1f} |")
    ratio = statistics.median(ctx_tok) / statistics.median(ctx_chars) if ctx_chars else 0
    if ratio:
        delta = (0.25 / ratio - 1) * 100
        verb = "underestimates" if delta < 0 else "overestimates"
        print(f"\ntokens per character (median contexts): {ratio:.3f}   ->  chars/4 {verb} tokens by {abs(delta):.0f}%")
    print(f"Cached per-question counts: {cache_path}")


if __name__ == "__main__":
    main()
