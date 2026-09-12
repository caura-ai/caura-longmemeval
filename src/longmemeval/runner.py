"""Evaluation runner for LongMemEval."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from datetime import datetime, timezone
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
        concurrency: int = 5,
    ) -> dict[str, Any]:
        items_by_id = {it.question_id: it for it in items}
        eval_results: list[EvaluationResult] = []
        qtype_results: dict[str, list[bool]] = {}

        def judge_entry(h: HypothesisEntry) -> EvaluationResult | None:
            ref = items_by_id.get(h.question_id)
            if not ref:
                return None

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
            return EvaluationResult(
                question_id=h.question_id,
                question=ref.question,
                gold_answer=ref.answer,
                hypothesis=h.hypothesis,
                question_type=qtype,
                correct=is_correct,
                judge_reason=reason,
                judge_model=getattr(self.judge_llm, "model_name", "unknown"),
            )

        if concurrency > 1 and len(hypotheses) > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                res_list = list(pool.map(judge_entry, hypotheses))
            eval_results = [r for r in res_list if r is not None]
        else:
            for h in hypotheses:
                r = judge_entry(h)
                if r is not None:
                    eval_results.append(r)

        for res in eval_results:
            qtype_results.setdefault(res.question_type, []).append(res.correct)

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


def run_reader_pipeline(
    reader_llm: BaseLLM,
    question: str,
    context: str,
    question_date: str | None = None,
    pipeline: str = "direct",
) -> tuple[str, float, dict[str, Any] | None]:
    """Execute either direct generation or multi-stage agentic-v1 pipeline."""
    t0_gen = time.perf_counter()

    if pipeline == "agentic-v1":
        # Stage 1: Extract factual evidence and classify support status
        t0_stage = time.perf_counter()
        evidence_bundle = reader_llm.extract_evidence(
            question=question,
            context=context,
            question_date=question_date,
        )
        extract_ms = (time.perf_counter() - t0_stage) * 1000

        # Stage 2: Generate initial candidate from extracted evidence
        t0_stage = time.perf_counter()
        initial_answer = reader_llm.answer_from_evidence(
            question=question,
            evidence=evidence_bundle,
            question_date=question_date,
        )
        initial_ms = (time.perf_counter() - t0_stage) * 1000

        # Stage 3: Conditional narrow inference or temporal calculation
        inference_answer: str | None = None
        inference_ms = 0.0
        should_infer = (
            evidence_bundle.status == "inferable"
            or "not mentioned" in initial_answer.lower()
            or "might" in question.lower()
            or "how many" in question.lower()
        )
        if should_infer and evidence_bundle.facts:
            t0_stage = time.perf_counter()
            inference_answer = reader_llm.infer_answer(
                question=question,
                evidence=evidence_bundle,
                question_date=question_date,
            )
            inference_ms = (time.perf_counter() - t0_stage) * 1000
        elif (evidence_bundle.status == "unsupported" or not evidence_bundle.facts) and context.strip():
            # If evidence extractor returned unsupported on a non-empty context,
            # generate candidate from raw context so verifier can arbitrate
            t0_stage = time.perf_counter()
            direct_prompt = build_answer_prompt(
                query=question,
                context=context,
                question_date=question_date,
            )
            inference_answer = reader_llm.generate(direct_prompt)
            inference_ms = (time.perf_counter() - t0_stage) * 1000

        # Stage 4: Verify and, if unsupported, hold back
        candidates = (initial_answer,) + ((inference_answer,) if inference_answer else ())
        t0_stage = time.perf_counter()
        verified = reader_llm.verify_answer(
            question=question,
            evidence=evidence_bundle,
            candidates=candidates,
            question_date=question_date,
        )
        verify_ms = (time.perf_counter() - t0_stage) * 1000

        if not evidence_bundle.facts and inference_answer:
            # If initial fact extraction returned empty on non-empty context,
            # fall back to direct context-grounded candidate
            hypothesis_ans = inference_answer
            verifier_reason = f"Fact extractor returned empty; adopted direct context candidate. Verifier note: {verified.reason}"
        else:
            hypothesis_ans = verified.answer
            verifier_reason = verified.reason

        pipeline_trace = {
            "pipeline": "agentic-v1",
            "evidence_status": evidence_bundle.status,
            "support_status": evidence_bundle.status,
            "facts": list(evidence_bundle.facts),
            "extracted_facts": list(evidence_bundle.facts),
            "requirements": list(evidence_bundle.requirements),
            "initial_answer": initial_answer,
            "inference_answer": inference_answer,
            "verifier_reason": verifier_reason,
            "extract_windows": evidence_bundle.windows,
            "extract_ms": extract_ms,
            "initial_ms": initial_ms,
            "inference_ms": inference_ms,
            "verify_ms": verify_ms,
        }
    else:
        prompt = build_answer_prompt(
            query=question,
            context=context,
            question_date=question_date,
        )
        hypothesis_ans = reader_llm.generate(prompt)
        pipeline_trace = None

    gen_ms = (time.perf_counter() - t0_gen) * 1000
    return hypothesis_ans, gen_ms, pipeline_trace


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
        pipeline: str = "direct",
        exclude_ids: set[str] | list[str] | None = None,
        seed: int | None = None,
        concurrency: int = 1,
    ) -> dict[str, Any]:
        items = self.dataset.load_items(
            category=category,
            limit=limit,
            limit_per_category=limit_per_category,
            question_id=question_id,
            exclude_ids=exclude_ids,
            seed=seed,
        )
        if not items:
            console.print("[red]No questions found matching criteria.[/red]")
            return {}

        effective_name = run_name or f"{self.provider.name}-{pipeline}-{int(time.time())}"
        run_dir = self.output_dir / effective_name
        run_dir.mkdir(parents=True, exist_ok=True)
        hypotheses_path = run_dir / "hypotheses.jsonl"
        eval_path = run_dir / "eval_results.json"
        started_at = datetime.now(timezone.utc)

        console.print(
            f"\n[bold]Starting LongMemEval Benchmark[/bold]\n"
            f"Provider: [cyan]{self.provider.name}[/cyan] | Pipeline: [cyan]{pipeline}[/cyan] | Questions: [cyan]{len(items)}[/cyan] | Run: [cyan]{effective_name}[/cyan]\n"
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

        items_to_process = [it for it in items if it.question_id not in existing_qids]
        file_lock = threading.Lock()
        completed_count = len(existing_qids)
        total_items = len(items)

        def process_item(item: LongMemEvalItem) -> HypothesisEntry:
            nonlocal completed_count
            unit_id = item.question_id
            docs = self.dataset.item_to_documents(item)

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

            # 3. Generate answer
            context_text = format_facts(facts)
            hypothesis_ans, gen_ms, pipeline_trace = run_reader_pipeline(
                reader_llm=self.reader_llm,
                question=item.question,
                context=context_text,
                question_date=item.question_date,
                pipeline=pipeline,
            )

            entry = HypothesisEntry(
                question_id=item.question_id,
                hypothesis=hypothesis_ans,
                question=item.question,
                answer=item.answer,
                question_type=item.question_type,
                context=context_text,
                retrieve_time_ms=retrieve_ms,
                generate_time_ms=gen_ms,
                pipeline=pipeline,
                pipeline_trace=pipeline_trace,
            )

            with file_lock:
                completed_count += 1
                hypotheses.append(entry)
                with open(hypotheses_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
                console.print(f"  [{completed_count}/{total_items}] #{item.question_id} ({item.question_type}) in {retrieve_ms:.0f}ms ret / {gen_ms:.0f}ms gen")

            return entry

        if concurrency > 1 and len(items_to_process) > 1:
            console.print(f"[bold cyan]Processing {len(items_to_process)} questions with concurrency={concurrency}...[/bold cyan]")
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(process_item, items_to_process))
        else:
            for item in items_to_process:
                console.print(f"[bold blue]Question {item.question_id} ({item.question_type})[/bold blue]")
                process_item(item)

        # 4. Evaluate using official judge prompts
        console.print("\n[bold]Evaluating hypotheses with judge LLM...[/bold]")
        summary = self.evaluator.evaluate(hypotheses, items, out_eval_path=eval_path, concurrency=max(1, concurrency))

        finished_at = datetime.now(timezone.utc)
        duration_s = (finished_at - started_at).total_seconds()

        # Build comprehensive benchmark result payload and generate stunning HTML report
        from .report import build_report_payload, render_report

        run_meta = {
            "name": effective_name,
            "provider": self.provider.name,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_s,
            "top_k": top_k,
            "skip_ingest": skip_ingest,
            "reader": getattr(self.reader_llm, "model_name", "unknown"),
            "judge": getattr(self.judge_llm, "model_name", "unknown"),
            "pipeline": pipeline,
            "parameters": {
                "selection": {
                    "category": category,
                    "limit": limit,
                    "limit_per_category": limit_per_category,
                    "question_id": question_id,
                    "total_questions": len(items),
                    "seed": seed,
                    "excluded_count": len(exclude_ids) if exclude_ids else 0,
                },
                "ingestion": {
                    "provider": self.provider.name,
                    "skip_ingest": skip_ingest,
                    "chunk_chars": getattr(self.provider, "chunk_chars", None),
                    "chunk_mode": getattr(self.provider, "chunk_mode", None),
                    "agent_prefix": getattr(self.provider, "agent_prefix", None),
                },
                "retrieval": {
                    "top_k": top_k,
                    "category_adaptive": getattr(self.provider, "category_adaptive", None),
                    "sibling_expansion": getattr(self.provider, "sibling_expansion", None),
                    "context_budget_chars": getattr(self.provider, "context_budget_chars", None),
                    "sibling_window": getattr(self.provider, "sibling_window", None),
                    "as_of_recall": getattr(self.provider, "send_valid_at", False),
                    "valid_at": getattr(self.provider, "send_valid_at", False),
                    "context_ordering": "Chronological (oldest to newest)",
                },
                "generation": {
                    "pipeline": pipeline,
                    "reader": getattr(self.reader_llm, "model_name", "unknown"),
                    "judge": getattr(self.judge_llm, "model_name", "unknown"),
                    "prompt": "agentic-v1 (extract -> answer -> infer -> verify)" if pipeline == "agentic-v1" else "Official answer prompt with chronological context",
                },
            },
        }

        result_payload = build_report_payload(run_meta, summary, hypotheses)

        # Save consolidated results.json
        results_path = run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)

        # Render HTML report
        report_path = render_report(result_payload, run_dir / "report.html")

        # Print rich summary table
        table = Table(title=f"LongMemEval Results: {effective_name}")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Total", justify="right")

        for qtype, stats in summary["by_question_type"].items():
            table.add_row(qtype, f"{stats['accuracy']:.1%}", str(stats["total"]))

        table.add_row("OVERALL", f"[bold green]{summary['overall_accuracy']:.1%}[/bold green]", str(summary["total_questions"]))
        console.print(table)

        console.print(f"\n[bold green]HTML Benchmark Report:[/bold green] [cyan]{report_path}[/cyan]\n")

        return summary
