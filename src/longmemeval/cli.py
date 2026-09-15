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
from .runner import (
    DEFAULT_PRIMARY_JUDGE,
    DEFAULT_SECONDARY_JUDGE,
    JUDGE_PROTOCOL,
    BenchmarkRunner,
    Evaluator,
    run_reader_pipeline,
)
from .report import secondary_eval_filename, summarize_secondary
from .models import HypothesisEntry

app = typer.Typer(help="LongMemEval benchmark harness for Caura.ai")
console = Console()


def resolve_judges(
    judge_llm: str | None,
    judge_model: str | None,
    secondary_judge_llm: str | None,
    secondary_judge_model: str | None,
    no_secondary: bool,
) -> tuple[tuple[str, str], tuple[str, str] | None]:
    """Judge protocol: primary = headline (default gpt-4o, the LongMemEval reference judge);
    secondary = strict development judge (default gemini-3.5-flash-lite), always run beside the primary.

    Resolution order for each slot: CLI flag > JUDGE_LLM/JUDGE_MODEL (SECONDARY_JUDGE_LLM/SECONDARY_JUDGE_MODEL)
    env > protocol default. A model given without a provider inherits the default provider for that slot.
    """
    p_llm = judge_llm or os.environ.get("JUDGE_LLM") or DEFAULT_PRIMARY_JUDGE[0]
    p_model = judge_model or os.environ.get("JUDGE_MODEL") or (DEFAULT_PRIMARY_JUDGE[1] if p_llm == DEFAULT_PRIMARY_JUDGE[0] else None)
    primary = (p_llm, p_model)
    if no_secondary:
        return primary, None
    s_llm = secondary_judge_llm or os.environ.get("SECONDARY_JUDGE_LLM") or DEFAULT_SECONDARY_JUDGE[0]
    s_model = secondary_judge_model or os.environ.get("SECONDARY_JUDGE_MODEL") or (DEFAULT_SECONDARY_JUDGE[1] if s_llm == DEFAULT_SECONDARY_JUDGE[0] else None)
    secondary = (s_llm, s_model)
    if secondary == primary:
        return primary, None
    return primary, secondary


def build_judges(
    judge_llm: str | None,
    judge_model: str | None,
    secondary_judge_llm: str | None,
    secondary_judge_model: str | None,
    no_secondary: bool,
):
    primary, secondary = resolve_judges(judge_llm, judge_model, secondary_judge_llm, secondary_judge_model, no_secondary)
    primary_llm = get_llm(provider=primary[0], model=primary[1])
    secondary_llm = None
    if secondary is not None:
        try:
            secondary_llm = get_llm(provider=secondary[0], model=secondary[1])
        except Exception as exc:  # missing key for the secondary provider must not block the run
            console.print(f"[yellow]Secondary judge {secondary[0]}:{secondary[1]} unavailable ({str(exc)[:120]}); running primary only[/yellow]")
    console.print(
        f"[dim]Judges: primary {getattr(primary_llm, 'model_name', primary[1])}"
        + (f", secondary {getattr(secondary_llm, 'model_name', secondary[1])}" if secondary_llm else ", no secondary")
        + "[/dim]"
    )
    return primary_llm, secondary_llm


JUDGE_HELP = "Primary judge provider (headline): gemini | openai | grok. Default gpt-4o via openai (LongMemEval reference judge)"
JUDGE_MODEL_HELP = "Primary judge model name"
SECONDARY_JUDGE_HELP = "Secondary judge provider (reported beside the primary, never instead). Default gemini-3.5-flash-lite"
SECONDARY_JUDGE_MODEL_HELP = "Secondary judge model name"
NO_SECONDARY_HELP = "Run the primary judge only"
CONTEXT_FORMAT_HELP = (
    "Layout of the retrieved context handed to the reader: full (one block per stored chunk with its "
    "header and date/type trailer, the published layout) | compact (same stored text grouped by session, "
    "one header per session, no per-chunk bookkeeping, '(in reply to)' repeats dropped)"
)


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
    judge_llm: Optional[str] = typer.Option(None, "--judge", help=JUDGE_HELP),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help=JUDGE_MODEL_HELP),
    secondary_judge_llm: Optional[str] = typer.Option(None, "--secondary-judge", help=SECONDARY_JUDGE_HELP),
    secondary_judge_model: Optional[str] = typer.Option(None, "--secondary-judge-model", help=SECONDARY_JUDGE_MODEL_HELP),
    no_secondary_judge: bool = typer.Option(False, "--no-secondary-judge", help=NO_SECONDARY_HELP),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip document ingestion (use existing store)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Retrieval top_k (max 200 for Caura). Default: provider profile (chars mode 20 + category profiles; turns mode flat 50)"),
    as_of_recall: bool = typer.Option(True, "--as-of-recall/--no-as-of-recall", help="Enable As-Of Recall: anchor temporal ranking and valid_at at question_date"),
    chunk_mode: Optional[str] = typer.Option(None, "--chunk-mode", help="Caura ingestion chunking: chars (4k parts, default) | turns (one memory per user statement + reply)"),
    chunk_chars: Optional[int] = typer.Option(None, "--chunk-chars", help="Max chunk size in characters (default 4000 for chars, 1200 for turns)"),
    sibling_expansion: Optional[bool] = typer.Option(None, "--sibling-expansion/--no-sibling-expansion", help="After ranking, pull the other chunks of each hit's session into the context (budgeted)"),
    context_budget: Optional[int] = typer.Option(None, "--context-budget", help="Character budget for the retrieved context when sibling expansion is on (default 150000)"),
    sibling_window: Optional[int] = typer.Option(None, "--sibling-window", help="Neighbouring chunks per side to pull around each hit (default 0 = whole session)"),
    raw_turns_only: Optional[bool] = typer.Option(None, "--raw-turns-only/--allow-derived", help="Drop server-derived memories (no metadata.doc_id) from /search results so the reader sees only stored source text (default on in turns mode)"),
    agent_prefix: Optional[str] = typer.Option(None, "--agent-prefix", help="Caura agent-id prefix; use a fresh prefix to ingest into a separate store without touching an existing one"),
    bulk_size: Optional[int] = typer.Option(None, "--bulk-size", help="Items per /memories/bulk call (max 100); raise for fine-grained chunking"),
    pipeline: str = typer.Option("direct", "--pipeline", help="Pipeline architecture: direct | agentic-v1"),
    context_format: str = typer.Option("full", "--context-format", help=CONTEXT_FORMAT_HELP),
    exclude_results: Optional[list[Path]] = typer.Option(None, "--exclude-results", help="Exclude question IDs from previous results.json / eval_results.json / hypotheses.jsonl"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for sampling questions"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent workers for retrieval, generation, and judging"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
):
    """Run LongMemEval benchmark with selected memory provider and evaluator."""
    ds = LongMemEvalDataset(data_path=data_path)
    provider_kwargs: dict = {"send_valid_at": as_of_recall, "as_of_recall": as_of_recall}
    if chunk_mode is not None:
        provider_kwargs["chunk_mode"] = chunk_mode
    if chunk_chars is not None:
        provider_kwargs["chunk_chars"] = chunk_chars
    if sibling_expansion is not None:
        provider_kwargs["sibling_expansion"] = sibling_expansion
    if context_budget is not None:
        provider_kwargs["context_budget_chars"] = context_budget
    if sibling_window is not None:
        provider_kwargs["sibling_window"] = sibling_window
    if raw_turns_only is not None:
        provider_kwargs["raw_turns_only"] = raw_turns_only
    if agent_prefix is not None:
        provider_kwargs["agent_prefix"] = agent_prefix
    if bulk_size is not None:
        provider_kwargs["bulk_size"] = bulk_size
    mem_provider = get_memory_provider(provider, **provider_kwargs)
    reader = get_llm(provider=reader_llm, model=reader_model or os.environ.get("READER_MODEL"))
    judge, secondary_judge = build_judges(judge_llm, judge_model, secondary_judge_llm, secondary_judge_model, no_secondary_judge)

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
        secondary_judge_llm=secondary_judge,
        context_format=context_format,
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
def ingest(
    provider: str = typer.Option("caura", "--provider", "-p", help=f"Memory provider: {list(PROVIDERS.keys())}"),
    category: Optional[str] = typer.Option(None, "--category", help=f"Filter question category: {QUESTION_TYPES}"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit total questions"),
    limit_per_category: Optional[int] = typer.Option(None, "--per-category", help="Limit questions per category"),
    question_id: Optional[str] = typer.Option(None, "--question-id", "-q", help="Single question ID"),
    name: str = typer.Option(..., "--name", help="Marker directory name under outputs/ (progress is recorded in <name>/ingested.txt for resume)"),
    chunk_mode: Optional[str] = typer.Option(None, "--chunk-mode", help="chars | turns"),
    chunk_chars: Optional[int] = typer.Option(None, "--chunk-chars", help="Max chunk size in characters"),
    agent_prefix: Optional[str] = typer.Option(None, "--agent-prefix", help="Caura agent-id prefix for this store"),
    bulk_size: Optional[int] = typer.Option(None, "--bulk-size", help="Items per /memories/bulk call (max 100)"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Concurrent questions being ingested"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for sampling questions"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
):
    """Ingest haystacks only (no retrieval/generation), so a later `run --skip-ingest` reads a settled store."""
    import concurrent.futures
    import threading
    import time as _time

    ds = LongMemEvalDataset(data_path=data_path)
    provider_kwargs: dict = {}
    if chunk_mode is not None:
        provider_kwargs["chunk_mode"] = chunk_mode
    if chunk_chars is not None:
        provider_kwargs["chunk_chars"] = chunk_chars
    if agent_prefix is not None:
        provider_kwargs["agent_prefix"] = agent_prefix
    if bulk_size is not None:
        provider_kwargs["bulk_size"] = bulk_size
    mem_provider = get_memory_provider(provider, **provider_kwargs)
    # No point sleeping per question here; the store settles while the rest ingests.
    if hasattr(mem_provider, "settle_time"):
        mem_provider.settle_time = 0.0

    items = ds.load_items(
        category=category, limit=limit, limit_per_category=limit_per_category, question_id=question_id, seed=seed
    )
    marker_dir = output_dir / name
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / "ingested.txt"
    done: set[str] = set()
    if marker.exists():
        done = {l.strip() for l in marker.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in items if it.question_id not in done]
    console.print(f"[bold cyan]Ingesting {len(todo)} questions ({len(done)} already done) with concurrency={concurrency}...[/bold cyan]")

    lock = threading.Lock()
    counter = {"n": len(done)}

    failed: list[str] = []

    def one(item):
        docs = ds.item_to_documents(item)
        t0 = _time.perf_counter()
        try:
            mem_provider.reset_unit(item.question_id)
            stored = mem_provider.ingest(item.question_id, docs)
        except Exception as exc:  # keep going; unmarked questions are re-ingested on resume
            with lock:
                failed.append(item.question_id)
                console.print(f"  [red]#{item.question_id} failed: {str(exc)[:200]}[/red]")
            return
        dt = _time.perf_counter() - t0
        with lock:
            counter["n"] += 1
            with open(marker, "a", encoding="utf-8") as f:
                f.write(item.question_id + "\n")
            console.print(f"  [{counter['n']}/{len(items)}] #{item.question_id}: {len(docs)} sessions -> {stored} chunks in {dt:.1f}s")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(one, todo))
    finally:
        mem_provider.cleanup()
    if failed:
        console.print(f"[yellow]{len(failed)} question(s) failed and are not marked done; rerun to retry: {', '.join(failed)}[/yellow]")
    console.print(f"[green]Done. Progress marker: {marker}[/green]")


@app.command()
def rerun_pipeline(
    source_hypotheses: Path = typer.Argument(..., help="Path to existing hypotheses.jsonl file with saved context"),
    run_name: str = typer.Option(..., "--name", "-n", help="Name for the new run output directory"),
    pipeline: str = typer.Option("agentic-v1", "--pipeline", help="Pipeline architecture: direct | agentic-v1"),
    reader_llm: str = typer.Option("gemini", "--reader", help="LLM for answering: gemini | openai | grok"),
    reader_model: Optional[str] = typer.Option(None, "--reader-model", help="Model name for reader LLM"),
    judge_llm: Optional[str] = typer.Option(None, "--judge", help=JUDGE_HELP),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help=JUDGE_MODEL_HELP),
    secondary_judge_llm: Optional[str] = typer.Option(None, "--secondary-judge", help=SECONDARY_JUDGE_HELP),
    secondary_judge_model: Optional[str] = typer.Option(None, "--secondary-judge-model", help=SECONDARY_JUDGE_MODEL_HELP),
    no_secondary_judge: bool = typer.Option(False, "--no-secondary-judge", help=NO_SECONDARY_HELP),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent workers for reader and judge LLMs"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    skip_generation: bool = typer.Option(False, "--skip-generation", help="Skip generation if hypotheses.jsonl already exists"),
    generate_html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML benchmark report"),
    context_format: str = typer.Option("full", "--context-format", help=CONTEXT_FORMAT_HELP + " Saved contexts are re-laid-out offline; the reformatted context is what gets saved."),
):
    """Re-run answer generation and evaluation on already-retrieved context from an earlier run."""
    import json
    import concurrent.futures
    from datetime import datetime, timezone
    from .context_format import CONTEXT_FORMATS, reformat_context
    from .report import generate_report_for_run

    if context_format not in CONTEXT_FORMATS:
        raise typer.BadParameter(f"--context-format must be one of {CONTEXT_FORMATS}")

    out_run_dir = output_dir / run_name
    out_run_dir.mkdir(parents=True, exist_ok=True)

    ds = LongMemEvalDataset(data_path=data_path)
    items_by_id = {item.question_id: item for item in ds.load_items()}

    reader = get_llm(provider=reader_llm, model=reader_model or os.environ.get("READER_MODEL"))
    judge, secondary_judge = build_judges(judge_llm, judge_model, secondary_judge_llm, secondary_judge_model, no_secondary_judge)

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
            context = reformat_context(src.context or "", context_format)

            hypothesis_ans, gen_ms, pipeline_trace, reader_usage = run_reader_pipeline(
                reader_llm=reader,
                question=src.question,
                context=context,
                question_date=q_date,
                pipeline=pipeline,
            )

            entry = HypothesisEntry(
                question_id=src.question_id,
                hypothesis=hypothesis_ans,
                question=src.question,
                answer=src.answer,
                question_type=src.question_type,
                context=context,
                retrieve_time_ms=src.retrieve_time_ms,
                generate_time_ms=gen_ms,
                pipeline=pipeline,
                pipeline_trace=pipeline_trace,
                retrieval_stats=src.retrieval_stats,
                reader_usage=reader_usage,
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

    judge_name = getattr(judge, "model_name", "unknown")
    console.print(f"\n[bold]Evaluating with primary judge ({judge_name})...[/bold]")
    evaluator = Evaluator(judge_llm=judge)
    eval_path = out_run_dir / "eval_results.json"
    eval_summary = evaluator.evaluate(
        hypotheses=new_hypotheses,
        items=list(items_by_id.values()),
        out_eval_path=eval_path,
        concurrency=concurrency,
    )
    secondary_block = None
    secondary_name = None
    if secondary_judge is not None:
        secondary_name = getattr(secondary_judge, "model_name", "unknown")
        console.print(f"[bold]Evaluating with secondary judge ({secondary_name})...[/bold]")
        secondary_path = out_run_dir / secondary_eval_filename(secondary_name)
        try:
            secondary_summary = Evaluator(judge_llm=secondary_judge).evaluate(
                hypotheses=new_hypotheses,
                items=list(items_by_id.values()),
                out_eval_path=secondary_path,
                concurrency=concurrency,
            )
            secondary_block = summarize_secondary(secondary_name, secondary_path.name, eval_summary, secondary_summary)
            console.print(
                f"[dim]{judge_name}: {eval_summary['correct_questions']}/{eval_summary['total_questions']}  |  "
                f"{secondary_name}: {secondary_summary['correct_questions']}/{secondary_summary['total_questions']}  |  "
                f"agreement {secondary_block['agreement_with_primary']}[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]Secondary judge failed: {str(exc)[:200]}[/yellow]")

    # Save run metadata and results.json
    reader_name = f"{reader.__class__.__name__}:{getattr(reader, 'model_name', getattr(reader, 'model', 'default'))}"

    run_meta = {
        "name": run_name,
        "pipeline": pipeline,
        "reader": reader_name,
        "judge": judge_name,
        "judge_secondary": secondary_name,
        "judge_protocol": JUDGE_PROTOCOL,
        "started_at": start_time.isoformat(),
        "finished_at": finish_time.isoformat(),
        "duration_seconds": (finish_time - start_time).total_seconds(),
        "parameters": {
            "source_hypotheses": str(source_hypotheses),
            "pipeline": pipeline,
            "context_format": context_format,
            "generation": {
                "pipeline": pipeline,
                "reader": reader_name,
                "judge": judge_name,
                "judge_secondary": secondary_name,
            },
        },
    }
    # Store, retrieval and server are the source run's; only the reader side is new.
    from .report import inherit_source_run_meta

    inherit_source_run_meta(run_meta)
    payload: dict = {"run": run_meta, "summary": eval_summary}
    if secondary_block is not None:
        payload["secondary_evaluation"] = secondary_block
    with open(out_run_dir / "results.json", "w", encoding="utf-8") as f_res:
        json.dump(payload, f_res, indent=2, ensure_ascii=False)

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
    judge_llm: Optional[str] = typer.Option(None, "--judge", help=JUDGE_HELP),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help=JUDGE_MODEL_HELP),
    secondary_judge_llm: Optional[str] = typer.Option(None, "--secondary-judge", help=SECONDARY_JUDGE_HELP),
    secondary_judge_model: Optional[str] = typer.Option(None, "--secondary-judge-model", help=SECONDARY_JUDGE_MODEL_HELP),
    no_secondary_judge: bool = typer.Option(False, "--no-secondary-judge", help=NO_SECONDARY_HELP + " (implied when --output is given)"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write eval_results.json (single-judge mode)"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent judge calls"),
    generate_html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML benchmark report"),
):
    """Evaluate an existing hypothesis file against ground truth without running ingestion/retrieval.

    Default writes eval_results.json (primary judge) and eval_results_<secondary>.json next to the hypotheses.
    With --output only the primary judge runs and its verdicts go to that path (for ad-hoc judge comparisons).
    """
    import json
    from .report import generate_report_for_run

    ds = LongMemEvalDataset(data_path=data_path)
    items = ds.load_items()

    hypotheses: list[HypothesisEntry] = []
    with open(hypotheses_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hypotheses.append(HypothesisEntry.model_validate(json.loads(line)))

    single = no_secondary_judge or output_path is not None
    judge, secondary_judge = build_judges(judge_llm, judge_model, secondary_judge_llm, secondary_judge_model, single)
    eval_target = output_path or (hypotheses_path.parent / "eval_results.json")
    summary = Evaluator(judge_llm=judge).evaluate(hypotheses, items, out_eval_path=eval_target, concurrency=concurrency)
    console.print(f"{getattr(judge, 'model_name', 'primary')}: {summary['correct_questions']}/{summary['total_questions']} ({summary['overall_accuracy']:.1%})")

    if secondary_judge is not None:
        secondary_name = getattr(secondary_judge, "model_name", "unknown")
        secondary_path = hypotheses_path.parent / secondary_eval_filename(secondary_name)
        try:
            s2 = Evaluator(judge_llm=secondary_judge).evaluate(hypotheses, items, out_eval_path=secondary_path, concurrency=concurrency)
            block = summarize_secondary(secondary_name, secondary_path.name, summary, s2)
            console.print(
                f"{secondary_name}: {s2['correct_questions']}/{s2['total_questions']} ({s2['overall_accuracy']:.1%})  |  "
                f"agreement with primary {block['agreement_with_primary']}/{summary['total_questions']}"
            )
            results_json = hypotheses_path.parent / "results.json"
            if results_json.exists():
                try:
                    payload = json.loads(results_json.read_text(encoding="utf-8"))
                    payload["secondary_evaluation"] = block
                    payload.setdefault("run", {})["judge_secondary"] = secondary_name
                    payload["run"]["judge_protocol"] = JUDGE_PROTOCOL
                    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as exc:
                    console.print(f"[yellow]Could not update results.json with secondary evaluation: {exc}[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Secondary judge failed: {str(exc)[:200]}[/yellow]")

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
