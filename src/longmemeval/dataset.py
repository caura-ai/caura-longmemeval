"""LongMemEval dataset downloader and loader."""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .models import LongMemEvalItem, MemoryDocument

console = Console()

# Official HuggingFace dataset file URL (cleaned LongMemEval-S)
DATA_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_s_cleaned.json"
)

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
]


def get_default_data_dir() -> Path:
    """Return local data directory."""
    root = Path(__file__).resolve().parents[2]
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_dataset(target_path: Path | None = None) -> Path:
    """Ensure LongMemEval dataset is downloaded and return path."""
    if target_path is None:
        target_path = get_default_data_dir() / "longmemeval_s_cleaned.json"

    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    console.print(f"[bold cyan]Downloading LongMemEval dataset to {target_path}...[/bold cyan]")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress report
    urllib.request.urlretrieve(DATA_URL, target_path)
    console.print(f"[bold green]Downloaded successfully ({target_path.stat().st_size / (1024*1024):.1f} MB).[/bold green]")
    return target_path


def parse_timestamp(date_str: str) -> datetime | None:
    """Parse LongMemEval date string format into timezone-aware datetime."""
    if not date_str:
        return None
    try:
        # Strip day-of-week e.g. "(Sat)", "(Wed)" while preserving time:
        # "2023/05/20 (Sat) 02:21" -> "2023/05/20 02:21"
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", str(date_str)).strip()
        for fmt in [
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ]:
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except Exception:
        return None


def extract_question_ids_from_file(file_path: Path | str) -> set[str]:
    """Extract question IDs from an existing results.json, eval_results.json, or hypotheses.jsonl."""
    p = Path(file_path)
    if not p.exists():
        return set()
    qids: set[str] = set()
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if "question_id" in d:
                            qids.add(d["question_id"])
                    except Exception:
                        pass
        return qids

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "questions" in data and isinstance(data["questions"], list):
                qids.update(q["question_id"] for q in data["questions"] if isinstance(q, dict) and "question_id" in q)
            elif "results" in data and isinstance(data["results"], list):
                qids.update(q["question_id"] for q in data["results"] if isinstance(q, dict) and "question_id" in q)
    except Exception:
        pass
    return qids


class LongMemEvalDataset:
    """Loader and adapter for LongMemEval."""

    def __init__(self, data_path: Path | str | None = None):
        if data_path is not None:
            self.data_path = Path(data_path)
        else:
            env_path = os.environ.get("LONGMEMEVAL_DATA_PATH")
            if env_path:
                self.data_path = Path(env_path)
            else:
                self.data_path = get_default_data_dir() / "longmemeval_s_cleaned.json"

    def ensure_data(self) -> Path:
        if not self.data_path.exists():
            download_dataset(self.data_path)
        return self.data_path

    def load_items(
        self,
        category: str | None = None,
        limit: int | None = None,
        limit_per_category: int | None = None,
        question_id: str | None = None,
        exclude_ids: set[str] | list[str] | None = None,
        seed: int | None = None,
    ) -> list[LongMemEvalItem]:
        """Load benchmark questions with optional filtering, category balancing, exclusion, and deterministic seeding."""
        if hasattr(self, "_items") and self._items is not None:
            raw = [item.model_dump() for item in self._items]
        else:
            path = self.ensure_data()
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

        allowed_categories = None
        if category:
            allowed_categories = {c.strip() for c in category.split(",") if c.strip()}

        excluded_set = set(exclude_ids) if exclude_ids else set()

        raw_candidates: list[dict[str, Any]] = []
        for d in raw:
            qid = d.get("question_id", "")
            qtype = d.get("question_type", "")

            if question_id and qid != question_id:
                continue
            if allowed_categories and qtype not in allowed_categories:
                continue
            if excluded_set and qid in excluded_set:
                continue

            raw_candidates.append(d)

        # Deterministic shuffle if seed is provided
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(raw_candidates)

        items: list[LongMemEvalItem] = []
        cat_counts: dict[str, int] = {}

        for d in raw_candidates:
            qid = d.get("question_id", "")
            qtype = d.get("question_type", "")

            if limit_per_category is not None:
                if cat_counts.get(qtype, 0) >= limit_per_category:
                    continue

            item = LongMemEvalItem(
                question_id=qid,
                question=str(d.get("question", "")),
                answer=str(d.get("answer", "")),
                question_type=qtype,
                question_date=str(d.get("question_date", "")),
                haystack_sessions=d.get("haystack_sessions", []),
                haystack_dates=d.get("haystack_dates", []),
                haystack_session_ids=d.get("haystack_session_ids", []),
                other_attributes={k: v for k, v in d.items() if k not in {
                    "question_id", "question", "answer", "question_type",
                    "question_date", "haystack_sessions", "haystack_dates", "haystack_session_ids"
                }},
            )
            items.append(item)
            cat_counts[qtype] = cat_counts.get(qtype, 0) + 1

            if limit and len(items) >= limit:
                break

        return items

    def item_to_documents(self, item: LongMemEvalItem) -> list[MemoryDocument]:
        """Convert an item's haystack sessions into MemoryDocuments."""
        docs: list[MemoryDocument] = []
        sessions = item.haystack_sessions
        dates = item.haystack_dates
        sids = item.haystack_session_ids

        min_len = min(len(sessions), len(dates), len(sids))
        for sess, d_str, sid in zip(sessions[:min_len], dates[:min_len], sids[:min_len]):
            doc_id = f"{item.question_id}_{sid}"
            lines: list[str] = []
            for t in sess:
                if not isinstance(t, dict):
                    continue
                role = str(t.get("role", "user")).strip().capitalize()
                content = str(t.get("content", "")).strip()
                if content:
                    lines.append(f"{role}: {content}")
            session_text = "\n\n".join(lines)

            dt = parse_timestamp(d_str)
            dt_iso = dt.isoformat() if dt else None
            date_display = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "unknown"
            ctx = f"Session {doc_id} - happened on {date_display} UTC."

            docs.append(
                MemoryDocument(
                    id=doc_id,
                    content=session_text,
                    user_id=item.question_id,
                    timestamp=dt_iso,
                    context=ctx,
                )
            )
        return docs
