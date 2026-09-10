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
from .runner import BenchmarkRunner, Evaluator
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
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Local path to longmemeval_s_cleaned.json"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o", help="Directory for benchmark outputs"),
):
    """Run LongMemEval benchmark with selected memory provider and evaluator."""
    ds = LongMemEvalDataset(data_path=data_path)
    mem_provider = get_memory_provider(provider)
    reader = get_llm(provider=reader_llm, model=reader_model or os.environ.get("READER_MODEL"))
    judge = get_llm(provider=judge_llm, model=judge_model or os.environ.get("JUDGE_MODEL"))

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
        )
    finally:
        mem_provider.cleanup()


@app.command()
def evaluate_hypotheses(
    hypotheses_path: Path = typer.Argument(..., help="Path to hypotheses.jsonl file"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="Path to reference longmemeval_s_cleaned.json"),
    judge_llm: str = typer.Option("gemini", "--judge", help="Judge provider: gemini | openai | grok"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model name for judge LLM"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write eval_results.json"),
):
    """Evaluate an existing hypothesis file against ground truth without running ingestion/retrieval."""
    import json
    ds = LongMemEvalDataset(data_path=data_path)
    items = ds.load_items()

    hypotheses: list[HypothesisEntry] = []
    with open(hypotheses_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hypotheses.append(HypothesisEntry.model_validate(json.loads(line)))

    judge = get_llm(provider=judge_llm, model=judge_model or os.environ.get("JUDGE_MODEL"))
    evaluator = Evaluator(judge_llm=judge)
    evaluator.evaluate(hypotheses, items, out_eval_path=output_path)


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
