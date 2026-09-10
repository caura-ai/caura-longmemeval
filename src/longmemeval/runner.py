"""Evaluation runner for LongMemEval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .dataset import LongMemEvalDataset
from .llm import BaseLLM, get_llm
from .models import EvaluationResult, HypothesisEntry, LongMemEvalItem
from .prompts import build_answer_prompt, get_official_judge_prompt
from .providers.base import BaseMemoryProvider

console = Console()


def format_facts(facts) -> str:
    """Format retrieved memory facts into context text for answer prompt, sorted chronologically."""
    if not facts:
        return ""
    # Sort facts chronologically (oldest to newest) by timestamp if available
    sorted_facts = sorted(
        facts,
        key=lambda f: (0, f.timestamp) if f.timestamp else (1, ""),
    )
    lines = []
    for f in sorted_facts:
        chunk = []
        if f.title:
            chunk.append(f.title)
        chunk.append(f.content)
        if f.timestamp:
            chunk.append(f"date: {f.timestamp}")
        if f.memory_type:
            chunk.append(f"type: {f.memory_type}")
        if f.tags:
            chunk.append(f"tags: {', '.join(f.tags)}")
        lines.append("\n".join(chunk))
    return "\n---\n".join(lines)


class Evaluator:
    """Evaluates hypothesis JSONL files against reference dataset using LLM judges."""

    def __init__(self, judge_llm: BaseLLM):
        self.judge_llm = judge_llm

    def evaluate(
        self,
        hypotheses: list[HypothesisEntry],
        items: list[LongMemEvalItem],
        out_eval_path: Path | None = None,
    ) -> dict[str, Any]:
        items_by_id = {it.question_id: it for it in items}
        eval_results: list[EvaluationResult] = []
        qtype_results: dict[str, list[bool]] = {}

        for h in hypotheses:
            ref = items_by_id.get(h.question_id)
            if not ref:
                continue

            qtype = ref.question_type
            is_abstention = "_abs" in h.question_id
            judge_prompt = get_official_judge_prompt(
                task=qtype,
                question=ref.question,
                answer=ref.answer,
                response=h.hypothesis,
                abstention=is_abstention,
            )

            is_correct, reason = self.judge_llm.judge_bool(judge_prompt)
            eval_results.append(
                EvaluationResult(
                    question_id=h.question_id,
                    question=ref.question,
                    gold_answer=ref.answer,
                    hypothesis=h.hypothesis,
                    question_type=qtype,
                    correct=is_correct,
                    judge_reason=reason,
                    judge_model=getattr(self.judge_llm, "model_name", "unknown"),
                )
            )

            qtype_results.setdefault(qtype, []).append(is_correct)

        total = len(eval_results)
        total_correct = sum(1 for r in eval_results if r.correct)
        overall_acc = (total_correct / total) if total > 0 else 0.0

        qtype_acc = {}
        for qt, scores in qtype_results.items():
            acc = sum(scores) / len(scores) if scores else 0.0
            qtype_acc[qt] = {"accuracy": acc, "total": len(scores), "correct": sum(scores)}

        summary = {
            "overall_accuracy": overall_acc,
            "total_questions": total,
            "correct_questions": total_correct,
            "by_question_type": qtype_acc,
            "results": [r.model_dump() for r in eval_results],
        }

        if out_eval_path:
            out_eval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_eval_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved evaluation results to {out_eval_path}[/green]")

        return summary


class BenchmarkRunner:
    """Executes the full LongMemEval pipeline: Ingest -> Retrieve -> Answer -> Evaluate."""

    def __init__(
        self,
        dataset: LongMemEvalDataset,
        provider: BaseMemoryProvider,
        reader_llm: BaseLLM,
        judge_llm: BaseLLM,
        output_dir: Path = Path("outputs"),
    ):
        self.dataset = dataset
        self.provider = provider
        self.reader_llm = reader_llm
        self.judge_llm = judge_llm
        self.output_dir = output_dir
        self.evaluator = Evaluator(judge_llm=judge_llm)

    def run(
        self,
        category: str | None = None,
        limit: int | None = None,
        limit_per_category: int | None = None,
        question_id: str | None = None,
        run_name: str | None = None,
        skip_ingest: bool = False,
        top_k: int = 20,
    ) -> dict[str, Any]:
        items = self.dataset.load_items(
            category=category,
            limit=limit,
            limit_per_category=limit_per_category,
            question_id=question_id,
        )
        if not items:
            console.print("[red]No questions found matching criteria.[/red]")
            return {}

        effective_name = run_name or f"{self.provider.name}-{int(time.time())}"
        run_dir = self.output_dir / effective_name
        run_dir.mkdir(parents=True, exist_ok=True)
        hypotheses_path = run_dir / "hypotheses.jsonl"
        eval_path = run_dir / "eval_results.json"

        console.print(
            f"\n[bold]Starting LongMemEval Benchmark[/bold]\n"
            f"Provider: [cyan]{self.provider.name}[/cyan] | Questions: [cyan]{len(items)}[/cyan] | Run: [cyan]{effective_name}[/cyan]\n"
        )

        hypotheses: list[HypothesisEntry] = []
        existing_qids: set[str] = set()
        if hypotheses_path.exists():
            with open(hypotheses_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = HypothesisEntry.model_validate(json.loads(line))
                            hypotheses.append(entry)
                            existing_qids.add(entry.question_id)
                        except Exception:
                            pass
            if existing_qids:
                console.print(f"[yellow]Resuming run: found {len(existing_qids)} already completed questions in {hypotheses_path}[/yellow]")

        for idx, item in enumerate(items, 1):
            if item.question_id in existing_qids:
                console.print(f"[dim][{idx}/{len(items)}] Skipping {item.question_id} ({item.question_type}) - already completed[/dim]")
                continue

            console.print(f"[bold blue][{idx}/{len(items)}][/bold blue] Question {item.question_id} ({item.question_type})")

            # 1. Ingest unit documents (haystack sessions)
            docs = self.dataset.item_to_documents(item)
            unit_id = item.question_id

            if not skip_ingest:
                t0_ingest = time.perf_counter()
                self.provider.reset_unit(unit_id)
                num_stored = self.provider.ingest(unit_id, docs)
                ingest_s = time.perf_counter() - t0_ingest
                console.print(f"  [dim]Ingested {len(docs)} sessions ({num_stored} stored facts/chunks) in {ingest_s:.2f}s[/dim]")

            # 2. Retrieve memories
            t0_ret = time.perf_counter()
            facts = self.provider.retrieve(
                unit_id=unit_id,
                query=item.question,
                top_k=top_k,
                query_date=item.question_date,
                question_type=item.question_type,
            )
            retrieve_ms = (time.perf_counter() - t0_ret) * 1000
            console.print(f"  [dim]Retrieved {len(facts)} memories in {retrieve_ms:.0f}ms[/dim]")

            # 3. Generate answer
            context_text = format_facts(facts)
            prompt = build_answer_prompt(
                query=item.question,
                context=context_text,
                question_date=item.question_date,
            )

            t0_gen = time.perf_counter()
            hypothesis_ans = self.reader_llm.generate(prompt)
            gen_ms = (time.perf_counter() - t0_gen) * 1000
            console.print(f"  [dim]Generated answer in {gen_ms:.0f}ms[/dim]")

            entry = HypothesisEntry(
                question_id=item.question_id,
                hypothesis=hypothesis_ans,
                question=item.question,
                answer=item.answer,
                question_type=item.question_type,
                context=context_text,
                retrieve_time_ms=retrieve_ms,
                generate_time_ms=gen_ms,
            )
            hypotheses.append(entry)

            # Save incrementally
            with open(hypotheses_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")

        # 4. Evaluate using official judge prompts
        console.print("\n[bold]Evaluating hypotheses with judge LLM...[/bold]")
        summary = self.evaluator.evaluate(hypotheses, items, out_eval_path=eval_path)

        # Print rich summary table
        table = Table(title=f"LongMemEval Results: {effective_name}")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Total", justify="right")

        for qtype, stats in summary["by_question_type"].items():
            table.add_row(qtype, f"{stats['accuracy']:.1%}", str(stats["total"]))

        table.add_row("OVERALL", f"[bold green]{summary['overall_accuracy']:.1%}[/bold green]", str(summary["total_questions"]))
        console.print(table)

        return summary
