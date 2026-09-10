"""CLI for LongMemEval Caura benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

# Load .env before other imports
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

from .dataset import LongMemEvalDataset, QUESTION_TYPES, download_dataset
from .llm import get_llm
from .providers import get_memory_provider, PROVIDERS
from .runner import BenchmarkRunner, Evaluator, run_reader_pipeline
from .models import HypothesisEntry

app = typer.Typer(help="LongMemEval benchmark harness for Caura.ai")
console = Console()


@app.command()
def download(
    data_path: Optional[Path] = typer.Option(
        None,
        "--data-path",
        "-d",
        help="Target path to save longmemeval_s_cleaned.json",
    ),
):
    """Download the LongMemEval dataset if not already downloaded."""
    target = download_dataset(data_path)
    console.print(f"Dataset ready at: {target}")


@app.command()
def run(
    provider: str = typer.Option("caura", "--provider", "-p", help=f"Memory provider: {list(PROVIDERS.keys())}"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help=f"Filter question category (comma-separated or single): {QUESTION_TYPES}"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit total questions to evaluate"),
    limit_per_category: Optional[int] = typer.Option(None, "--per-category", help="Limit questions per category (useful for balanced evaluation)"),
    question_id: Optional[str] = typer.Option(None, "--question-id", "-q", help="Evaluate single question ID"),
    run_name: Optional[str] = typer.Option(None, "--name", help="Custom name for the run output directory"),
    reader_llm: str = typer.Option("gemini", "--reader", help="LLM for answering: gemini | openai | grok"),
    reader_model: Optional[str] = typer.Option(None, "--reader-model", help="Model name for reader LLM"),
    judge_llm: str = typer.Option("gemini", "--judge", help="LLM for judging: gemini | openai | grok"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model name for judge LLM"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip document ingestion (use existing store)"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Retrieval top_k (max 200 for Caura; when category adaptive is active, uses category profiles)"),
    pipeline: str = typer.Option("direct", "--pipeline", help="Pipeline architecture: direct | agentic-v1"),
    exclude_results: Optional[list[Path]] = typer.Option(None, "--exclude-results", help="Exclude question IDs from previous results.json / eval_results.json / hypotheses.jsonl"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for sampling questions"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent workers for retrieval, generation, and judging"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
):
    """Run LongMemEval benchmark with selected memory provider and evaluator."""
    ds = LongMemEvalDataset(data_path=data_path)
    mem_provider = get_memory_provider(provider)
    reader = get_llm(provider=reader_llm, model=reader_model or os.environ.get("READER_MODEL"))
    judge = get_llm(provider=judge_llm, model=judge_model or os.environ.get("JUDGE_MODEL"))

    excluded_ids: set[str] = set()
    if exclude_results:
        from .dataset import extract_question_ids_from_file
        for p in exclude_results:
            excluded_ids.update(extract_question_ids_from_file(p))
        if excluded_ids:
            console.print(f"[yellow]Excluding {len(excluded_ids)} question IDs from evaluation[/yellow]")

    runner = BenchmarkRunner(
        dataset=ds,
        provider=mem_provider,
        reader_llm=reader,
        judge_llm=judge,
        output_dir=output_dir,
    )

    try:
        runner.run(
            category=category,
            limit=limit,
            limit_per_category=limit_per_category,
            question_id=question_id,
            run_name=run_name,
            skip_ingest=skip_ingest,
            top_k=top_k,
            pipeline=pipeline,
            exclude_ids=excluded_ids if excluded_ids else None,
            seed=seed,
            concurrency=concurrency,
        )
    finally:
        mem_provider.cleanup()


@app.command()
def rerun_pipeline(
    source_hypotheses: Path = typer.Argument(..., help="Path to existing hypotheses.jsonl file with saved context"),
    run_name: str = typer.Option(..., "--name", "-n", help="Name for the new run output directory"),
    pipeline: str = typer.Option("agentic-v1", "--pipeline", help="Pipeline architecture: direct | agentic-v1"),
    reader_llm: str = typer.Option("gemini", "--reader", help="LLM for answering: gemini | openai | grok"),
    reader_model: Optional[str] = typer.Option(None, "--reader-model", help="Model name for reader LLM"),
    judge_llm: str = typer.Option("gemini", "--judge", help="LLM for judging: gemini | openai | grok"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model name for judge LLM"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent workers for reader and judge LLMs"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip generation if hypotheses.jsonl already exists"),
    generate_html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML benchmark report"),
):
    """Re-run answer generation and evaluation on already-retrieved context from an earlier run."""
    import json
    import concurrent.futures
    from datetime import datetime, timezone
    from .report import generate_report_for_run

    out_run_dir = output_dir / run_name
    out_run_dir.mkdir(parents=True, exist_ok=True)

    ds = LongMemEvalDataset(data_path=data_path)
    items_by_id = {item.question_id: item for item in ds.load_items()}

    reader = get_llm(provider=reader_llm, model=reader_model or os.environ.get("READER_MODEL"))
    judge = get_llm(provider=judge_llm, model=judge_model or os.environ.get("JUDGE_MODEL"))

    # Load source hypotheses and deduplicate by question_id
    hypos_dict: dict[str, HypothesisEntry] = {}
    with open(source_hypotheses, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = HypothesisEntry.model_validate(json.loads(line))
                hypos_dict[entry.question_id] = entry
    source_entries: list[HypothesisEntry] = list(hypos_dict.values())

    new_hypotheses: list[HypothesisEntry] = []
    hypo_path = out_run_dir / "hypotheses.jsonl"
    start_time = datetime.now(timezone.utc)

    if skip_generation and hypo_path.exists():
        console.print(f"[bold cyan]Using existing {hypo_path} (--skip-generation)...[/bold cyan]")
        with open(hypo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    new_hypotheses.append(HypothesisEntry.model_validate(json.loads(line)))
    else:
        console.print(f"[bold cyan]Re-running {len(source_entries)} questions with pipeline='{pipeline}' (concurrency={concurrency})...[/bold cyan]")
        new_hypotheses = [None] * len(source_entries)  # type: ignore

        def process_one(idx_src: tuple[int, HypothesisEntry]) -> tuple[int, HypothesisEntry]:
            idx, src = idx_src
            item = items_by_id.get(src.question_id)
            q_date = item.question_date if item else None

            hypothesis_ans, gen_ms, pipeline_trace = run_reader_pipeline(
                reader_llm=reader,
                question=src.question,
                context=src.context,
                question_date=q_date,
                pipeline=pipeline,
            )

            entry = HypothesisEntry(
                question_id=src.question_id,
                hypothesis=hypothesis_ans,
                question=src.question,
                answer=src.answer,
                question_type=src.question_type,
                context=src.context,
                retrieve_time_ms=src.retrieve_time_ms,
                generate_time_ms=gen_ms,
                pipeline=pipeline,
                pipeline_trace=pipeline_trace,
            )
            return idx, entry

        if concurrency > 1 and len(source_entries) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {
                    executor.submit(process_one, (i, src)): (i, src)
                    for i, src in enumerate(source_entries)
                }
                completed_count = 0
                for fut in concurrent.futures.as_completed(future_map):
                    i, src = future_map[fut]
                    idx, entry = fut.result()
                    new_hypotheses[idx] = entry
                    completed_count += 1
                    console.print(f"  [{completed_count}/{len(source_entries)}] #{entry.question_id} ({entry.question_type}) generated in {entry.generate_time_ms:.0f}ms")
        else:
            for idx, src in enumerate(source_entries):
                console.print(f"[{idx+1}/{len(source_entries)}] #{src.question_id} ({src.question_type})")
                _, entry = process_one((idx, src))
                new_hypotheses[idx] = entry
                console.print(f"  [dim]Generated answer ({pipeline}) in {entry.generate_time_ms:.0f}ms[/dim]")

        with open(hypo_path, "w", encoding="utf-8") as f_out:
            for entry in new_hypotheses:
                f_out.write(entry.model_dump_json() + "\n")

    finish_time = datetime.now(timezone.utc)

    console.print("\n[bold]Evaluating with judge LLM...[/bold]")
    evaluator = Evaluator(judge_llm=judge)
    eval_path = out_run_dir / "eval_results.json"
    eval_summary = evaluator.evaluate(
        hypotheses=new_hypotheses,
        items=list(items_by_id.values()),
        out_eval_path=eval_path,
        concurrency=concurrency,
    )

    # Save run metadata and results.json
    reader_name = f"{reader.__class__.__name__}:{getattr(reader, 'model_name', getattr(reader, 'model', 'default'))}"
    judge_name = f"{judge.__class__.__name__}:{getattr(judge, 'model_name', getattr(judge, 'model', 'default'))}"

    run_meta = {
        "name": run_name,
        "pipeline": pipeline,
        "reader": reader_name,
        "judge": judge_name,
        "started_at": start_time.isoformat(),
        "finished_at": finish_time.isoformat(),
        "duration_seconds": (finish_time - start_time).total_seconds(),
        "parameters": {
            "source_hypotheses": str(source_hypotheses),
            "pipeline": pipeline,
            "generation": {
                "pipeline": pipeline,
                "reader": reader_name,
                "judge": judge_name,
            },
        },
    }
    with open(out_run_dir / "results.json", "w", encoding="utf-8") as f_res:
        json.dump({"run": run_meta, "summary": eval_summary}, f_res, indent=2, ensure_ascii=False)

    if generate_html:
        try:
            report_path = generate_report_for_run(out_run_dir)
            console.print(f"\n[bold green]HTML Benchmark Report:[/bold green] [cyan]{report_path}[/cyan]\n")
        except Exception as e:
            console.print(f"[yellow]Could not generate HTML report: {e}[/yellow]")


@app.command()
def evaluate_hypotheses(
    hypotheses_path: Path = typer.Argument(..., help="Path to hypotheses.jsonl file"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Path to reference longmemeval_s_cleaned.json"),
    judge_llm: str = typer.Option("gemini", "--judge", help="Judge provider: gemini | openai | grok"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model name for judge LLM"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write eval_results.json"),
    generate_html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML benchmark report"),
):
    """Evaluate an existing hypothesis file against ground truth without running ingestion/retrieval."""
    import json
    from .report import generate_report_for_run

    ds = LongMemEvalDataset(data_path=data_path)
    items = ds.load_items()

    hypotheses: list[HypothesisEntry] = []
    with open(hypotheses_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hypotheses.append(HypothesisEntry.model_validate(json.loads(line)))

    judge = get_llm(provider=judge_llm, model=judge_model or os.environ.get("JUDGE_MODEL"))
    evaluator = Evaluator(judge_llm=judge)
    eval_target = output_path or (hypotheses_path.parent / "eval_results.json")
    summary = evaluator.evaluate(hypotheses, items, out_eval_path=eval_target)

    if generate_html and eval_target:
        try:
            run_dir = eval_target.parent
            rep = generate_report_for_run(run_dir, eval_file_name=eval_target.name)
            console.print(f"\n[bold green]HTML Benchmark Report:[/bold green] [cyan]{rep}[/cyan]\n")
        except Exception as e:
            console.print(f"[yellow]Could not generate HTML report: {e}[/yellow]")


@app.command()
def report(
    run_path: Optional[str] = typer.Argument(
        None,
        help="Name or path of run directory in outputs/ (e.g. outputs/caura-50-adaptive-v4 or caura-50-adaptive-v4)",
    ),
    all_runs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Generate HTML reports for all run directories in outputs/",
    ),
    eval_file: str = typer.Option(
        "eval_results.json",
        "--eval-file",
        help="Name of evaluation results file (e.g. eval_results.json or eval_results_gpt4o.json)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Custom destination path for the HTML report",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the generated HTML report in your default browser",
    ),
):
    """Generate a stunning HTML benchmark report for one or all runs."""
    import webbrowser
    from .report import generate_report_for_run

    outputs_dir = Path("outputs")
    target_runs: list[Path] = []

    if all_runs:
        if not outputs_dir.exists():
            console.print("[red]Outputs directory does not exist.[/red]")
            raise typer.Exit(1)
        target_runs = [d for d in sorted(outputs_dir.iterdir()) if d.is_dir() and list(d.glob("eval_results*.json"))]
    elif run_path:
        p = Path(run_path)
        if not p.exists() and (outputs_dir / run_path).exists():
            p = outputs_dir / run_path
        if not p.exists():
            console.print(f"[red]Run directory does not exist: {run_path}[/red]")
            raise typer.Exit(1)
        target_runs = [p]
    else:
        # Default to the most recently modified run directory with eval_results
        if outputs_dir.exists():
            candidates = [d for d in outputs_dir.iterdir() if d.is_dir() and list(d.glob("eval_results*.json"))]
            if candidates:
                candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                target_runs = [candidates[0]]
                console.print(f"[dim]No run specified. Using latest run: {candidates[0].name}[/dim]")
        if not target_runs:
            console.print("[red]No runs found with evaluation results. Specify a run directory or use --all.[/red]")
            raise typer.Exit(1)

    last_report: Optional[Path] = None
    success_count = 0

    for rdir in target_runs:
        try:
            dest = output if (output and len(target_runs) == 1) else None
            rep = generate_report_for_run(rdir, output_file=dest, eval_file_name=eval_file)
            console.print(f"[bold green]Report rendered for {rdir.name}:[/bold green] [cyan]{rep}[/cyan]")
            last_report = rep
            success_count += 1
        except Exception as e:
            console.print(f"[yellow]Could not render report for {rdir.name}: {e}[/yellow]")

    console.print(f"\n[bold green]Successfully generated {success_count} HTML report(s).[/bold green]")

    if open_browser and last_report and last_report.exists():
        webbrowser.open(last_report.resolve().as_uri())


@app.command()
def stats(
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Path to longmemeval_s_cleaned.json"),
):
    """Display LongMemEval dataset statistics."""
    from collections import Counter
    from rich.table import Table

    ds = LongMemEvalDataset(data_path=data_path)
    items = ds.load_items()

    counts = Counter(it.question_type for it in items)
    total_sessions = sum(len(it.haystack_sessions) for it in items)

    table = Table(title="LongMemEval Dataset Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Total Questions", str(len(items)))
    table.add_row("Total Sessions", str(total_sessions))
    table.add_row("Avg Sessions / Question", f"{total_sessions / len(items):.1f}")
    table.add_section()

    for qtype in QUESTION_TYPES:
        table.add_row(qtype, str(counts.get(qtype, 0)))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
