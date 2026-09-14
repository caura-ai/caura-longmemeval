"""Caura (caura.ai / memclaw) memory provider for LongMemEval."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from rich.console import Console

from ..dataset import parse_timestamp
from ..models import MemoryDocument, RetrievedFact
from .base import BaseMemoryProvider

console = Console()

DEFAULT_BASE_URL = "https://caura.ai/api/v1"
MAX_CONTENT_LENGTH = 10_000
MAX_QUERY_LENGTH = 5_000
MAX_SEARCH_TOP_K = 200
BULK_MAX_ITEMS = 100
_BULK_MIN_INTERVAL_S = 0.55
RRF_K = 60
# Sibling expansion: cap on accepted hits per session (breadth over depth) and
# how many candidates to pull from /search when expansion is on.
MAX_SEEDS_PER_SESSION = 3
SIBLING_CANDIDATE_TOP_K = 150

CATEGORY_SEARCH_PROFILES: dict[str, dict[str, int]] = {
    "temporal-reasoning": {
        "top_k": 50,
        "merge_top_k": 50,
        "multiquery": 2,
    },
    "multi-session": {
        "top_k": 60,
        "merge_top_k": 60,
        "multiquery": 2,
    },
    "single-session-assistant": {
        "top_k": 40,
        "merge_top_k": 40,
        "multiquery": 2,
    },
    "knowledge-update": {
        "top_k": 30,
        "merge_top_k": 35,
        "multiquery": 2,
    },
    "single-session-user": {
        "top_k": 20,
        "merge_top_k": 20,
        "multiquery": 1,
    },
    "single-session-preference": {
        "top_k": 15,
        "merge_top_k": 15,
        "multiquery": 1,
    },
}

_QUERY_STOPWORDS = {
    "i'm", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "don", "should", "now", "could", "would", "tell",
    "remind", "checking", "previous", "chat", "remember", "wondering", "please", "ask", "answer",
}


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"CAURA_{name}", default)


def _sanitize_agent_id(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw)).strip("-") or "unit"
    if len(slug) > 180:
        slug = f"{slug[:170]}-{hashlib.sha1(str(raw).encode()).hexdigest()[:8]}"
    return slug


def _chunk_text(text: str, size: int = 4000) -> list[str]:
    """Split text on double newline (turns / paragraphs), falling back to line breaks and then slices."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            lines = para.split("\n")
            line_buf = ""
            for line in lines:
                if len(line) > size:
                    if line_buf:
                        chunks.append(line_buf)
                        line_buf = ""
                    for i in range(0, len(line), size):
                        chunks.append(line[i : i + size])
                else:
                    cand_line = f"{line_buf}\n{line}" if line_buf else line
                    if len(cand_line) > size:
                        chunks.append(line_buf)
                        line_buf = line
                    else:
                        line_buf = cand_line
            if line_buf:
                chunks.append(line_buf)
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > size:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


_ROLE_LINE_RE = re.compile(r"^(User|Assistant|System|Tool)\s*:", re.IGNORECASE)


def _chunk_turns(text: str, size: int = 1200) -> list[str]:
    """Split a session transcript into user-statement-centred chunks.

    The transcript is the ``Role: content`` blocks produced by
    ``LongMemEvalDataset.item_to_documents`` joined by blank lines. Each chunk
    starts at a User turn and carries the assistant reply that follows it, so a
    memory is "what the user said (and what they were told)". Replies longer
    than ``size`` are split and each piece is prefixed with the user turn it
    answers, so the embedding still centres on the user's statement.
    """
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return [text] if text else []

    # Group blocks into exchanges: [user_turn, reply_block, reply_block, ...]
    exchanges: list[list[str]] = []
    for block in blocks:
        is_user = bool(_ROLE_LINE_RE.match(block)) and block.split(":", 1)[0].strip().lower() == "user"
        if is_user or not exchanges:
            exchanges.append([block])
        else:
            exchanges[-1].append(block)

    chunks: list[str] = []
    for exchange in exchanges:
        joined = "\n\n".join(exchange)
        if len(joined) <= size:
            chunks.append(joined)
            continue

        user_turn = exchange[0]
        reply = "\n\n".join(exchange[1:])
        if len(user_turn) > size:
            chunks.extend(_chunk_text(user_turn, size))
        else:
            chunks.append(user_turn)
        if not reply:
            continue

        anchor = user_turn if len(user_turn) <= 240 else user_turn[:237].rstrip() + "..."
        prefix = f"(in reply to) {anchor}\n\n"
        reply_size = max(200, size - len(prefix))
        for piece in _chunk_text(reply, reply_size):
            chunks.append(prefix + piece)
    return chunks


class CauraMemoryProvider(BaseMemoryProvider):
    name = "caura"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        tenant_id: str | None = None,
        agent_prefix: str = "lme",
        ingest_mode: str = "bulk",
        chunk_chars: int | None = None,
        bulk_size: int | None = None,
        top_k: int | None = None,
        multiquery: int | None = None,
        merge_top_k: int | None = None,
        commit_batch: int = 50,
        ingest_workers: int = 4,
        category_adaptive: bool | None = None,
        send_valid_at: bool | None = None,
        as_of_recall: bool | None = None,
        chunk_mode: str | None = None,
        sibling_expansion: bool | None = None,
        context_budget_chars: int | None = None,
        sibling_window: int | None = None,
        raw_turns_only: bool | None = None,
    ):
        self.base_url = (base_url or _env("BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or _env("API_KEY")
        if not self.api_key:
            raise ValueError("CAURA_API_KEY environment variable is required")

        self._tenant_id = tenant_id or _env("TENANT_ID")
        self.agent_prefix = agent_prefix or _env("AGENT_PREFIX", "lme")
        self.ingest_mode = (ingest_mode or _env("INGEST", "bulk")).lower()
        # ``chars``: legacy 4k paragraph chunks (one session -> a few big parts).
        # ``turns``: one memory per user statement (+ its reply), see _chunk_turns.
        self.chunk_mode = (chunk_mode or _env("CHUNK_MODE", "chars")).lower()
        if self.chunk_mode not in ("chars", "turns"):
            raise ValueError(f"Unknown chunk_mode '{self.chunk_mode}' (expected 'chars' or 'turns')")
        if chunk_chars is not None:
            self.chunk_chars = chunk_chars
        elif self.chunk_mode == "turns":
            # CAURA_CHUNK_CHARS belongs to the legacy mode; turns has its own knob.
            self.chunk_chars = int(_env("TURN_CHUNK_CHARS", "1200"))
        else:
            self.chunk_chars = int(_env("CHUNK_CHARS", "4000"))

        # Sibling expansion: after ranking, pull the other chunks of every hit's
        # session (same doc_id in metadata) so a fact-bearing turn that ranked
        # low still reaches the reader when a sibling turn ranked high.
        if sibling_expansion is not None:
            self.sibling_expansion = bool(sibling_expansion)
        elif self.chunk_mode == "turns":
            self.sibling_expansion = _env("TURN_SIBLING_EXPANSION", "1").lower() in ("1", "true", "yes")
        else:
            self.sibling_expansion = _env("SIBLING_EXPANSION", "0").lower() in ("1", "true", "yes")
        if context_budget_chars is not None:
            self.context_budget_chars = int(context_budget_chars)
        else:
            self.context_budget_chars = int(_env("CONTEXT_BUDGET_CHARS", "150000"))
        # How many neighbouring chunks (each side) of a hit to pull from its
        # session. 0 = the whole session. On the balanced 54-set, whole-session
        # expansion gave gold-turn coverage 0.965 (51/54 fully covered) vs
        # 0.902 for a +-3 window and 0.913 for the legacy 4k-part retrieval,
        # so depth beats breadth here.
        if sibling_window is not None:
            self.sibling_window = max(0, int(sibling_window))
        else:
            self.sibling_window = max(0, int(_env("SIBLING_WINDOW", "0")))
        # The server derives its own memories (facts, preferences, tasks ...) from
        # what we write, asynchronously, with its own LLM. /search returns them
        # mixed with our stored chunks. ``raw_turns_only`` drops anything that is
        # not one of our chunks (no ``metadata.doc_id``) so the reader sees only
        # source text; how many were dropped is reported by ``pop_retrieval_stats``.
        if raw_turns_only is not None:
            self.raw_turns_only = bool(raw_turns_only)
        elif self.chunk_mode == "turns":
            self.raw_turns_only = _env("TURN_RAW_ONLY", "1").lower() in ("1", "true", "yes")
        else:
            self.raw_turns_only = _env("RAW_ONLY", "0").lower() in ("1", "true", "yes")
        self._retrieval_stats = threading.local()  # per worker thread; the runner is concurrent
        self._listing_supported: bool | None = None

        env_bulk = _env("BULK_SIZE")
        self.bulk_size = min(int(bulk_size if bulk_size is not None else (env_bulk or 25)), BULK_MAX_ITEMS)

        # Retrieval profile. Legacy (chars) mode keeps the category-adaptive
        # profiles and CAURA_TOP_K/MULTIQUERY/MERGE_TOP_K/CATEGORY_ADAPTIVE.
        # Turn mode defaults to a flat profile: top_k 50, single dense query,
        # no category logic. Measured on the balanced 54 with whole-session
        # expansion: flat k50/mq1 coverage 0.965 vs adaptive 0.961, and extra
        # keyword/broad query variants only lowered coverage (mq2 0.955,
        # mq3 0.936). Turn mode reads CAURA_TURN_* so the legacy .env values
        # for the 4k store do not leak in. Explicit args always win.
        turns = self.chunk_mode == "turns"
        env_key = (lambda k: f"TURN_{k}") if turns else (lambda k: k)
        mode_defaults = (
            {"TOP_K": "50", "MULTIQUERY": "1", "MERGE_TOP_K": "50", "CATEGORY_ADAPTIVE": "false"}
            if turns
            else {"TOP_K": "20", "MULTIQUERY": "2", "MERGE_TOP_K": "35", "CATEGORY_ADAPTIVE": "true"}
        )

        def knob(name: str) -> str:
            return _env(env_key(name), mode_defaults[name])  # type: ignore[return-value]

        if top_k is not None:
            self.top_k = min(int(top_k), MAX_SEARCH_TOP_K)
        else:
            self.top_k = min(int(knob("TOP_K")), MAX_SEARCH_TOP_K)

        if multiquery is not None:
            self.multiquery = multiquery
        else:
            self.multiquery = max(1, int(knob("MULTIQUERY")))

        if merge_top_k is not None:
            self.merge_top_k = merge_top_k
        else:
            self.merge_top_k = max(1, int(knob("MERGE_TOP_K")))

        if category_adaptive is not None:
            self.category_adaptive = category_adaptive
        else:
            self.category_adaptive = knob("CATEGORY_ADAPTIVE").lower() in ("true", "1", "yes")

        self.commit_batch = max(1, int(_env("COMMIT_BATCH", str(commit_batch))))
        self.ingest_workers = max(1, int(_env("INGEST_WORKERS", str(ingest_workers))))
        self.settle_time = float(_env("SETTLE_TIME", "5.0"))
        # Send the question's date as ``valid_at``. The runner has passed
        # ``query_date`` into retrieve() all along and this provider dropped it,
        # so temporal-reasoning questions were answered against today's date.
        # Default ON; CAURA_VALID_AT=0 reproduces earlier runs. Pair with
        # ``search.default_profile.freshness_reference=1`` on the tenant for the
        # freshness half (see the memory_bench adapter's docstring).
        if send_valid_at is not None:
            self.send_valid_at = bool(send_valid_at)
        elif as_of_recall is not None:
            self.send_valid_at = bool(as_of_recall)
        else:
            self.send_valid_at = _env("VALID_AT", "1").lower() in ("1", "true", "yes")

        self.as_of_recall = self.send_valid_at
        self._as_of_recall_ensured = False

        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        self._last_bulk_at = 0.0
        self.run_id = f"lme-{uuid.uuid4().hex[:8]}"

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(180.0, connect=30.0),
            )
        return self._client

    def _request(self, method: str, path: str, *, retries: int = 4, **kwargs) -> httpx.Response:
        delay = 1.0
        last: httpx.Response | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._http().request(method, path, **kwargs)
            except httpx.RequestError as exc:
                if attempt == retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last = resp
                if attempt == retries:
                    break
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                time.sleep(wait)
                delay *= 2
                continue
            return resp

        assert last is not None
        return last

    @property
    def tenant_id(self) -> str:
        if self._tenant_id:
            return self._tenant_id
        resp = self._request("GET", "/memories", params={"limit": 1})
        resp.raise_for_status()
        items = resp.json().get("items") or []
        if not items or not items[0].get("tenant_id"):
            raise RuntimeError("Could not auto-detect Caura tenant id. Please set CAURA_TENANT_ID.")
        self._tenant_id = items[0]["tenant_id"]
        return self._tenant_id

    def agent_id_for_unit(self, unit_id: str) -> str:
        return f"{self.agent_prefix}-{_sanitize_agent_id(unit_id)}"

    def reset_unit(self, unit_id: str) -> None:
        agent_id = self.agent_id_for_unit(unit_id)
        # Purge existing memories for this agent + fleet
        resp = self._request(
            "DELETE",
            "/memories",
            params={
                "tenant_id": self.tenant_id,
                "agent_id": agent_id,
                "fleet_id": agent_id,
            },
        )
        if resp.status_code not in (200, 204, 404):
            console.print(f"[yellow]Caura reset for {agent_id} returned {resp.status_code}: {resp.text[:150]}[/yellow]")

    def _bulk_write(self, agent_id: str, items: list[dict], mode: str = "strong") -> int:
        with self._lock:
            gap = time.monotonic() - self._last_bulk_at
            if gap < _BULK_MIN_INTERVAL_S:
                time.sleep(_BULK_MIN_INTERVAL_S - gap)
            self._last_bulk_at = time.monotonic()

        # Retries reuse the same X-Bulk-Attempt-Id, which the server uses to
        # recover items already committed before a 504/upstream timeout.
        resp = self._request(
            "POST",
            "/memories/bulk",
            retries=6,
            params={"mode": mode},
            json={
                "tenant_id": self.tenant_id,
                "agent_id": agent_id,
                "fleet_id": agent_id,
                "items": items,
            },
            headers={"X-Bulk-Attempt-Id": f"{self.run_id}-{uuid.uuid4().hex[:8]}"},
        )
        if resp.status_code not in (200, 207):
            raise RuntimeError(f"Caura bulk write failed [{resp.status_code}]: {resp.text[:400]}")
        body = resp.json()
        return body.get("created", 0)

    def _ingest_bulk(self, agent_id: str, documents: list[MemoryDocument]) -> int:
        items: list[dict] = []
        for doc in documents:
            if self.chunk_mode == "turns":
                chunks = _chunk_turns(doc.content, self.chunk_chars)
            else:
                chunks = _chunk_text(doc.content, self.chunk_chars)
            for idx, chunk in enumerate(chunks):
                header_bits = []
                if doc.timestamp:
                    header_bits.append(f"date: {doc.timestamp}")
                if doc.context:
                    header_bits.append(f"context: {doc.context}")
                if len(chunks) > 1:
                    unit = "turn" if self.chunk_mode == "turns" else "part"
                    header_bits.append(f"{unit} {idx+1}/{len(chunks)}")
                header = f"[{' | '.join(header_bits)}]\n" if header_bits else ""
                content = (header + chunk)[:MAX_CONTENT_LENGTH]

                items.append(
                    {
                        "content": content,
                        "source_uri": f"lme://{doc.id}",
                        "run_id": self.run_id,
                        "metadata": {
                            "bench": "longmemeval",
                            "doc_id": doc.id,
                            "chunk": idx,
                            "n_chunks": len(chunks),
                            "chunk_mode": self.chunk_mode,
                            **({"doc_timestamp": doc.timestamp} if doc.timestamp else {}),
                        },
                        **({"ts_valid_start": doc.timestamp} if doc.timestamp else {}),
                    }
                )

        created = 0
        for start in range(0, len(items), self.bulk_size):
            created += self._bulk_write(agent_id, items[start : start + self.bulk_size], mode="strong")
        return created

    def _ingest_extract_doc(self, agent_id: str, doc: MemoryDocument) -> int:
        header = f"Session date: {doc.timestamp}\n{doc.context}\n\n" if doc.timestamp else ""
        source_uri = f"lme://{doc.id}"

        resp = self._request(
            "POST",
            "/ingest/preview",
            json={
                "tenant_id": self.tenant_id,
                "agent_id": agent_id,
                "fleet_id": agent_id,
                "content": header + doc.content,
                "source_uri": source_uri,
            },
        )
        if resp.status_code != 200:
            return 0
        preview = resp.json()
        facts = preview.get("facts") or []
        if not facts:
            return 0

        created = 0
        for start in range(0, len(facts), self.commit_batch):
            batch = facts[start : start + self.commit_batch]
            commit = self._request(
                "POST",
                "/ingest/commit",
                json={
                    "tenant_id": self.tenant_id,
                    "agent_id": agent_id,
                    "fleet_id": agent_id,
                    "facts": batch,
                    "run_id": self.run_id,
                    **({"doc_hash": preview["doc_hash"]} if preview.get("doc_hash") else {}),
                },
            )
            if commit.status_code in (200, 201):
                created += int(commit.json().get("memories_created") or 0)
        return created

    def _ingest_extract(self, agent_id: str, documents: list[MemoryDocument]) -> int:
        total = 0
        with ThreadPoolExecutor(max_workers=self.ingest_workers) as pool:
            futures = [pool.submit(self._ingest_extract_doc, agent_id, doc) for doc in documents]
            for fut in as_completed(futures):
                try:
                    total += fut.result()
                except Exception:
                    pass
        return total

    def ingest(self, unit_id: str, documents: list[MemoryDocument]) -> int:
        agent_id = self.agent_id_for_unit(unit_id)
        if self.ingest_mode == "extract":
            stored = self._ingest_extract(agent_id, documents)
        else:
            stored = self._ingest_bulk(agent_id, documents)
        if self.settle_time > 0:
            time.sleep(self.settle_time)
        return stored

    def ensure_as_of_recall(self) -> bool:
        """Ensure tenant settings have freshness_reference=1 enabled on search.default_profile."""
        if self._as_of_recall_ensured:
            return True
        try:
            resp = self._request(
                "PUT",
                "/settings",
                json={"search": {"default_profile": {"freshness_reference": 1}}},
            )
            if resp.status_code in (200, 204):
                self._as_of_recall_ensured = True
                return True
        except Exception:
            pass
        return False

    def _search_once(
        self, query: str, agent_id: str, top_k: int | None = None, valid_at: str | None = None
    ) -> list[dict]:
        limit = min(top_k or self.top_k, MAX_SEARCH_TOP_K)
        body = {
            "tenant_id": self.tenant_id,
            "query": query[:MAX_QUERY_LENGTH],
            "top_k": limit,
            "filter_agent_id": agent_id,
        }
        if valid_at:
            body["valid_at"] = valid_at
        resp = self._request("POST", "/search", json=body)
        if resp.status_code == 422:
            retry_needed = False
            if limit > 20:
                body["top_k"] = 20
                retry_needed = True
            if "valid_at" in body:
                del body["valid_at"]
                retry_needed = True
            if retry_needed:
                resp = self._request("POST", "/search", json=body)
        if resp.status_code != 200:
            return []
        return resp.json().get("items") or []

    # ------------------------------------------------------------------ siblings

    def _list_agent_memories(self, agent_id: str, unit_id: str) -> list[dict]:
        """List every memory stored for one benchmark unit (agent), paging through /memories.

        Filters client-side on ``metadata.doc_id`` so the result is correct even
        if the server ignores the agent filter. Returns [] when listing is not
        supported so the caller can fall back to per-session search.
        """
        if self._listing_supported is False:
            return []
        page_size = 200
        max_pages = 40
        seen: dict[str, dict] = {}
        prefix = f"{unit_id}_"
        cursor: str | None = None
        offset = 0
        for _ in range(max_pages):
            params: dict = {
                "tenant_id": self.tenant_id,
                "agent_id": agent_id,
                "fleet_id": agent_id,
                "limit": page_size,
            }
            if cursor:
                params["cursor"] = cursor
            else:
                params["offset"] = offset
            resp = self._request("GET", "/memories", params=params)
            if resp.status_code != 200:
                if not seen:
                    self._listing_supported = False
                return list(seen.values())
            body = resp.json()
            items = body.get("items") or []
            for it in items:
                mid = str(it.get("id", ""))
                meta = it.get("metadata") or {}
                # The store also holds server-derived memories (no doc_id); skip them here.
                if mid and mid not in seen and str(meta.get("doc_id", "")).startswith(prefix):
                    seen[mid] = it
            if len(items) < page_size:
                break
            next_cursor = body.get("next_cursor")
            if next_cursor:
                if next_cursor == cursor:
                    break
                cursor = next_cursor
            else:
                offset += page_size
        if self._listing_supported is None:
            self._listing_supported = bool(seen)
        return list(seen.values())

    def _siblings_by_search(self, agent_id: str, doc_id: str, valid_at: str | None) -> list[dict]:
        """Fallback: find a session's chunks by searching for its header string.

        The header shows only the opaque session label (the part of ``doc_id``
        after the last underscore); the question id never reaches stored text.
        """
        label = doc_id.rsplit("_", 1)[-1]
        items = self._search_once(f"Session {label}", agent_id, top_k=60, valid_at=valid_at)
        return [it for it in items if (it.get("metadata") or {}).get("doc_id") == doc_id]

    def _expand_siblings(
        self,
        ranked: list[dict],
        agent_id: str,
        unit_id: str,
        seed_limit: int,
        valid_at: str | None,
    ) -> list[dict]:
        """Fill a character budget by walking hits in rank order.

        Every accepted hit brings its neighbourhood: the other chunks of the
        same session within ``sibling_window`` positions (or the whole session
        when the window is 0). At most ``seed_limit`` hits are accepted, and at
        most ``MAX_SEEDS_PER_SESSION`` per session so long sessions with many
        matching turns don't crowd out breadth. Stops when the budget is spent.
        Result is ordered (session date, session, chunk index) so the reader
        sees each session contiguously.
        """
        budget = self.context_budget_chars
        chosen: dict[str, dict] = {}
        chars = 0

        listing = self._list_agent_memories(agent_id, unit_id)
        by_doc: dict[str, list[dict]] = {}
        for it in listing:
            doc_id = str((it.get("metadata") or {}).get("doc_id", ""))
            if doc_id:
                by_doc.setdefault(doc_id, []).append(it)

        def chunk_idx(it: dict) -> int:
            return int((it.get("metadata") or {}).get("chunk", 0) or 0)

        def add(it: dict) -> bool:
            nonlocal chars
            mid = str(it.get("id", ""))
            if not mid or mid in chosen:
                return True
            size = len(it.get("content") or "")
            if chars + size > budget:
                return False
            chosen[mid] = it
            chars += size
            return True

        seeds_per_doc: dict[str, int] = {}
        fetched_fallback: dict[str, list[dict]] = {}
        seeds = 0
        for hit in ranked:
            if seeds >= seed_limit or chars >= budget:
                break
            mid = str(hit.get("id", ""))
            if not mid or mid in chosen:
                continue
            doc_id = str((hit.get("metadata") or {}).get("doc_id", ""))
            if doc_id and seeds_per_doc.get(doc_id, 0) >= MAX_SEEDS_PER_SESSION:
                continue
            if not add(hit):
                break
            seeds += 1
            if not doc_id:
                continue  # server-derived memory without a session; nothing to expand
            seeds_per_doc[doc_id] = seeds_per_doc.get(doc_id, 0) + 1

            sibs = by_doc.get(doc_id)
            if sibs is None and not listing:
                if doc_id not in fetched_fallback and len(fetched_fallback) < 12:
                    fetched_fallback[doc_id] = self._siblings_by_search(agent_id, doc_id, valid_at)
                sibs = fetched_fallback.get(doc_id, [])
            if not sibs:
                continue
            anchor = chunk_idx(hit)
            if self.sibling_window > 0:
                sibs = [s for s in sibs if abs(chunk_idx(s) - anchor) <= self.sibling_window]
            # Nearest neighbours first so a tight budget keeps the closest context.
            for sib in sorted(sibs, key=lambda s: (abs(chunk_idx(s) - anchor), chunk_idx(s))):
                if not add(sib):
                    break

        def sort_key(it: dict) -> tuple:
            meta = it.get("metadata") or {}
            ts = it.get("ts_valid_start") or meta.get("doc_timestamp") or it.get("created_at") or ""
            return (str(ts), str(meta.get("doc_id", "")), int(meta.get("chunk", 0) or 0))

        return sorted(chosen.values(), key=sort_key)

    def retrieve(
        self,
        unit_id: str,
        query: str,
        top_k: int = 20,
        query_date: str | None = None,
        question_type: str | None = None,
    ) -> list[RetrievedFact]:
        agent_id = self.agent_id_for_unit(unit_id)

        if self.send_valid_at and not self._as_of_recall_ensured:
            self.ensure_as_of_recall()

        target_top_k = top_k
        target_merge_top_k = self.merge_top_k
        target_multiquery = self.multiquery

        if self.category_adaptive and question_type and question_type in CATEGORY_SEARCH_PROFILES:
            profile = CATEGORY_SEARCH_PROFILES[question_type]
            target_top_k = profile.get("top_k", target_top_k)
            target_merge_top_k = profile.get("merge_top_k", target_merge_top_k)
            target_multiquery = profile.get("multiquery", target_multiquery)

        queries = [query]
        if target_multiquery > 1:
            content_words = [
                w for w in re.findall(r"[A-Za-z0-9'\-]+", query)
                if w.lower() not in _QUERY_STOPWORDS and len(w) > 2
            ]
            if content_words:
                queries.append(" ".join(content_words[:30]))
        if target_multiquery > 2:
            all_words = [w for w in re.findall(r"[A-Za-z0-9'\-]+", query) if len(w) > 2]
            if all_words and all_words != content_words:
                queries.append(" ".join(all_words[:30]))

        # LongMemEval's question_date is "2023/05/20 (Sat) 02:21"-shaped; the
        # dataset module already knows how to read it. Unparseable → no valid_at.
        valid_at: str | None = None
        if self.send_valid_at and query_date:
            parsed = parse_timestamp(query_date)
            valid_at = parsed.isoformat() if parsed else None

        # With sibling expansion the seed list is walked until the budget is
        # spent, so ask the server for a deeper candidate list.
        search_top_k = max(target_top_k, SIBLING_CANDIDATE_TOP_K) if self.sibling_expansion else target_top_k

        if len(queries) == 1:
            raw_items = self._search_once(queries[0], agent_id, top_k=search_top_k, valid_at=valid_at)
        else:
            fused: dict[str, float] = {}
            best: dict[str, dict] = {}
            for q in queries:
                for rank, item in enumerate(
                    self._search_once(q, agent_id, top_k=search_top_k, valid_at=valid_at)
                ):
                    mid = str(item.get("id"))
                    fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
                    best.setdefault(mid, item)
            raw_items = [best[mid] for mid in sorted(fused, key=lambda i: fused[i], reverse=True)]

        n_candidates = len(raw_items)
        n_derived = sum(1 for it in raw_items if not (it.get("metadata") or {}).get("doc_id"))
        if self.raw_turns_only:
            raw_items = [it for it in raw_items if (it.get("metadata") or {}).get("doc_id")]
        self._retrieval_stats.value = {
            "search_candidates": n_candidates,
            "server_derived_candidates": n_derived,
            "server_derived_dropped": n_derived if self.raw_turns_only else 0,
        }

        effective_limit = max(target_top_k, target_merge_top_k) if target_multiquery > 1 else target_top_k
        if self.sibling_expansion:
            # Budget-governed: walk the whole candidate list; the per-session
            # seed cap keeps breadth, the char budget bounds the context.
            selected = self._expand_siblings(raw_items, agent_id, unit_id, len(raw_items), valid_at)
        else:
            selected = raw_items[:effective_limit]

        results: list[RetrievedFact] = []
        for it in selected:
            meta = it.get("metadata") or {}
            ts = it.get("ts_valid_start") or meta.get("doc_timestamp") or it.get("created_at")
            results.append(
                RetrievedFact(
                    id=str(it.get("id", "")),
                    content=it.get("content", ""),
                    score=it.get("score"),
                    timestamp=ts,
                    memory_type=it.get("memory_type"),
                    title=it.get("title"),
                    tags=meta.get("tags") or [],
                )
            )
        return results

    def server_info(self) -> dict:
        """Server version and model configuration (GET /status), for run metadata."""
        try:
            resp = self._request("GET", "/status", retries=1)
            if resp.status_code != 200:
                return {"base_url": self.base_url, "status_code": resp.status_code}
            body = resp.json()
            search = {}
            settings = self._request("GET", "/settings", retries=1)
            if settings.status_code == 200:
                search = (settings.json().get("search") or {}).get("default_profile") or {}
            return {
                "base_url": self.base_url,
                "version": body.get("version"),
                "plugin_version": body.get("plugin_version"),
                "server_llm": body.get("llm"),
                "embedding": body.get("embedding"),
                "search_default_profile": search,
            }
        except Exception as exc:  # metadata only; never fail a run over it
            return {"base_url": self.base_url, "error": str(exc)[:200]}

    def pop_retrieval_stats(self) -> dict[str, int]:
        """Stats of the last ``retrieve`` on the calling thread (candidate and server-derived counts)."""
        stats = getattr(self._retrieval_stats, "value", None) or {}
        self._retrieval_stats.value = None
        return dict(stats)

    def cleanup(self, unit_id: str | None = None) -> None:
        if unit_id:
            try:
                self.reset_unit(unit_id)
            except Exception:
                pass
        if self._client is not None:
            self._client.close()
            self._client = None
