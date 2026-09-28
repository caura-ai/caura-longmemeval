"""Compare gold-turn coverage between two runs, question by question.

Retrieval is deterministic for a fixed store and server, so a coverage change
between two runs of the same configuration means the server changed what
``/search`` returns (upgrade, derivation filling the store, ranking change).
This script finds that in minutes without paying for a reader pass: point it at
two ``coverage.json`` files produced by ``scripts/coverage.py --json`` (or at two
run directories that contain one).

Usage:
    uv run python scripts/coverage_diff.py outputs/<run-a> outputs/<run-b> [--list]

Prints the total gold turns present in each run, the per-type coverage, how many
questions lost / gained gold turns, and, with ``--list``, every changed question
together with its per-question ``retrieval_stats`` from ``hypotheses.jsonl`` when
present (candidate counts, server-derived rows, store integrity).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_coverage(path: Path) -> dict:
    if path.is_dir():
        path = path / "coverage.json"
    if not path.exists():
        sys.exit(f"{path} not found; run scripts/coverage.py <run> --json <run>/coverage.json first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_retrieval_stats(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "hypotheses.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["question_id"]] = d.get("retrieval_stats") or {}
    return out


def per_type(questions: list[dict]) -> dict[str, tuple[int, int]]:
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for q in questions:
        if q.get("abstention"):
            continue
        acc[q["question_type"]][0] += int(q.get("gold_turns_present") or 0)
        acc[q["question_type"]][1] += int(q.get("gold_turns") or 0)
    return {k: (v[0], v[1]) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--list", action="store_true", help="Print every question whose coverage changed")
    args = ap.parse_args()

    a = load_coverage(args.run_a)
    b = load_coverage(args.run_b)
    qa = {q["question_id"]: q for q in a["questions"]}
    qb = {q["question_id"]: q for q in b["questions"]}
    common = [qid for qid in qa if qid in qb]
    if not common:
        sys.exit("The two runs share no question ids")

    def total(qs: dict, ids: list[str]) -> tuple[int, int]:
        present = sum(int(qs[i].get("gold_turns_present") or 0) for i in ids if not qs[i].get("abstention"))
        gold = sum(int(qs[i].get("gold_turns") or 0) for i in ids if not qs[i].get("abstention"))
        return present, gold

    pa, ga = total(qa, common)
    pb, gb = total(qb, common)
    name_a = args.run_a.name
    name_b = args.run_b.name
    print(f"questions compared: {len(common)}")
    print(f"gold turns present: {name_a} {pa}/{ga} ({pa / ga:.3f})   {name_b} {pb}/{gb} ({pb / gb:.3f})   delta {pb - pa:+d}")

    ta = per_type([qa[i] for i in common])
    tb = per_type([qb[i] for i in common])
    print("\nper type (present/gold):")
    for qt in sorted(set(ta) | set(tb)):
        xa, xb = ta.get(qt, (0, 0)), tb.get(qt, (0, 0))
        print(f"  {qt:28s} {xa[0]:4d}/{xa[1]:<4d} -> {xb[0]:4d}/{xb[1]:<4d}  ({xb[0] - xa[0]:+d})")

    lost = [i for i in common if int(qb[i].get("gold_turns_present") or 0) < int(qa[i].get("gold_turns_present") or 0)]
    gained = [i for i in common if int(qb[i].get("gold_turns_present") or 0) > int(qa[i].get("gold_turns_present") or 0)]
    print(f"\nquestions that lost gold turns: {len(lost)}   gained: {len(gained)}")

    fully_a = sum(1 for i in common if qa[i].get("fully_covered"))
    fully_b = sum(1 for i in common if qb[i].get("fully_covered"))
    print(f"fully covered questions: {fully_a} -> {fully_b}")

    stats_b = load_retrieval_stats(args.run_b if args.run_b.is_dir() else args.run_b.parent)
    if stats_b:
        keys = ("search_candidates", "server_derived_candidates", "injected_candidates",
                "store_derived_rows", "store_nonactive_raw", "store_unembedded_raw")
        print(f"\n{name_b} retrieval_stats medians over compared questions:")
        for k in keys:
            vals = sorted(float(stats_b[i].get(k)) for i in common if i in stats_b and isinstance(stats_b[i].get(k), (int, float)))
            if vals:
                print(f"  {k:28s} median {vals[len(vals) // 2]:g}  max {vals[-1]:g}")

    if args.list:
        print("\nchanged questions:")
        for i in sorted(lost + gained, key=lambda q: qa[q]["question_type"]):
            st = stats_b.get(i, {})
            extra = ""
            if st:
                extra = f" | cand {st.get('search_candidates')} derived {st.get('server_derived_candidates')} injected {st.get('injected_candidates', '-')} nonactive {st.get('store_nonactive_raw', '-')}"
            print(
                f"  {i:16s} {qa[i]['question_type']:28s} present {qa[i].get('gold_turns_present')}->{qb[i].get('gold_turns_present')} of {qa[i].get('gold_turns')}{extra}"
            )


if __name__ == "__main__":
    main()
