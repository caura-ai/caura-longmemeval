"""Gold-turn coverage and failure bucketing for a LongMemEval run.

For every non-abstention question, the dataset marks the turns that carry the
answer (``has_answer``) inside the gold sessions (``answer_session_ids``). This
script checks, for each of those turns, whether its text is present in the
context the reader saw (``hypotheses.jsonl``), and buckets every wrong answer as

- retrieval-bound: at least one gold turn is missing from the context
  (``partial`` if some gold turns are present, ``none`` if none are);
- reader-bound: every gold turn is present and the reader still answered wrong.

Coverage is measured at turn granularity on the raw text, so it is independent
of how the store chunked or labelled the sessions (4k parts, per-turn chunks,
opaque headers). A long assistant turn that the store split into several pieces
counts as covered when every line of it is present.

Usage:
    uv run python scripts/coverage.py outputs/<run> [--eval-file eval_results.json]
                                                    [--eval-file eval_results_gemini35flashlite.json]
                                                    [--json out.json] [--list-failures]

Multiple ``--eval-file`` values bucket the same contexts under several judges.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from longmemeval.dataset import LongMemEvalDataset  # noqa: E402

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", str(text)).strip().lower()


def turn_present(turn_text: str, ctx_norm: str) -> bool:
    """A gold turn is present if its whole text is in the context, or, when the store
    split it, if every non-trivial line of it is."""
    whole = norm(turn_text)
    if not whole:
        return True
    if whole in ctx_norm:
        return True
    lines = [norm(l) for l in str(turn_text).split("\n")]
    lines = [l for l in lines if len(l) >= 8]
    if not lines:
        return False
    for line in lines:
        if line in ctx_norm:
            continue
        # Very long single lines are sliced by the chunker; accept if sampled windows are present.
        if len(line) > 1000:
            step = max(1, (len(line) - 120) // 4)
            windows = [line[i : i + 120] for i in range(0, len(line) - 120 + 1, step)][:5]
            if windows and all(w in ctx_norm for w in windows):
                continue
        return False
    return True


def gold_turns(item) -> list[str]:
    gold_sids = set(item.other_attributes.get("answer_session_ids") or [])
    turns: list[str] = []
    for sess, sid in zip(item.haystack_sessions, item.haystack_session_ids):
        if sid not in gold_sids:
            continue
        for t in sess:
            if isinstance(t, dict) and t.get("has_answer") and str(t.get("content", "")).strip():
                turns.append(str(t["content"]))
    return turns


def load_hypotheses(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["question_id"]] = d  # last write wins (resumed runs may repeat)
    return out


def load_verdicts(path: Path) -> tuple[str, dict[str, bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    judge = results[0].get("judge_model", path.stem) if results else path.stem
    return judge, {r["question_id"]: bool(r["correct"]) for r in results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Run directory containing hypotheses.jsonl")
    ap.add_argument("--eval-file", action="append", default=None,
                    help="Judge verdict file(s) inside run_dir (default: eval_results.json and every eval_results_*.json present)")
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None, help="Write the full per-question table here")
    ap.add_argument("--list-failures", action="store_true", help="Print every failure with its bucket")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists() and (ROOT / "outputs" / run_dir).exists():
        run_dir = ROOT / "outputs" / run_dir
    hyps = load_hypotheses(run_dir / "hypotheses.jsonl")

    eval_files = args.eval_file or sorted(p.name for p in run_dir.glob("eval_results*.json"))
    verdict_sets = [load_verdicts(run_dir / name) for name in eval_files]

    ds = LongMemEvalDataset(data_path=args.data_path)
    items = {it.question_id: it for it in ds.load_items()}

    rows: list[dict] = []
    for qid, h in hyps.items():
        item = items.get(qid)
        if item is None:
            continue
        is_abs = qid.endswith("_abs")
        turns = gold_turns(item)
        ctx_norm = norm(h.get("context") or "")
        present = [turn_present(t, ctx_norm) for t in turns]
        row = {
            "question_id": qid,
            "question_type": item.question_type,
            "abstention": is_abs,
            "gold_turns": len(turns),
            "gold_turns_present": sum(present),
            "coverage": (sum(present) / len(turns)) if turns else None,
            "fully_covered": all(present) if turns else None,
            "context_chars": len(h.get("context") or ""),
        }
        for judge, verdicts in verdict_sets:
            row[f"correct[{judge}]"] = verdicts.get(qid)
        rows.append(row)

    scored = [r for r in rows if not r["abstention"] and r["gold_turns"] > 0]
    cov = [r["coverage"] for r in scored]
    print(f"Run: {run_dir}")
    print(f"Questions: {len(rows)} total, {len(scored)} non-abstention with gold turns, "
          f"{sum(1 for r in rows if r['abstention'])} abstention")
    print(f"Gold-turn coverage (mean over non-abstention): {statistics.mean(cov):.3f}")
    print(f"Fully covered questions: {sum(1 for r in scored if r['fully_covered'])}/{len(scored)}")
    print(f"Gold turns present: {sum(r['gold_turns_present'] for r in scored)}/{sum(r['gold_turns'] for r in scored)}")

    by_type: dict[str, list[float]] = {}
    for r in scored:
        by_type.setdefault(r["question_type"], []).append(r["coverage"])
    print("\nCoverage by category:")
    for qt, vals in sorted(by_type.items()):
        print(f"  {qt:28s} {statistics.mean(vals):.3f}  (n={len(vals)})")

    summary: dict = {
        "run": str(run_dir),
        "coverage_mean": statistics.mean(cov),
        "fully_covered": sum(1 for r in scored if r["fully_covered"]),
        "scored": len(scored),
        "by_judge": {},
    }
    for judge, verdicts in verdict_sets:
        key = f"correct[{judge}]"
        wrong = [r for r in scored if r.get(key) is False]
        wrong_abs = [r for r in rows if r["abstention"] and r.get(key) is False]
        partial = [r for r in wrong if 0 < r["gold_turns_present"] < r["gold_turns"]]
        none_ = [r for r in wrong if r["gold_turns_present"] == 0]
        reader = [r for r in wrong if r["fully_covered"]]
        n_correct = sum(1 for r in rows if r.get(key) is True)
        print(f"\nJudge {judge}: {n_correct}/{len(rows)} correct")
        print(f"  non-abstention failures: {len(wrong)}")
        print(f"    retrieval-bound (gold turn missing): {len(partial) + len(none_)}  (partial {len(partial)} / none {len(none_)})")
        print(f"    reader-bound (all gold turns present): {len(reader)}")
        print(f"  abstention failures: {len(wrong_abs)}")
        summary["by_judge"][judge] = {
            "correct": n_correct,
            "failures_non_abstention": len(wrong),
            "retrieval_bound": len(partial) + len(none_),
            "retrieval_bound_partial": len(partial),
            "retrieval_bound_none": len(none_),
            "reader_bound": len(reader),
            "abstention_failures": len(wrong_abs),
        }
        if args.list_failures:
            for r in sorted(wrong, key=lambda r: (r["fully_covered"], r["question_type"], r["question_id"])):
                bucket = "reader" if r["fully_covered"] else ("retrieval/none" if r["gold_turns_present"] == 0 else "retrieval/partial")
                print(f"    {r['question_id']:16s} {r['question_type']:26s} {bucket:18s} gold {r['gold_turns_present']}/{r['gold_turns']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "questions": rows}, indent=1), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
