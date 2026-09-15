"""Context layouts for the reader.

``full`` is the original layout: every retrieved chunk as stored (its own
``[date | context | turn i/n]`` header line, then the text), followed by the
provider's ``date:`` / ``type:`` / ``tags:`` trailer, blocks separated by ``---``.

``compact`` carries the same stored text with the bookkeeping removed. Chunks are
grouped by session and ordered by turn, each session gets one header (label and
date), and a split reply's ``(in reply to) <user turn>`` prefix is dropped when
that user turn is already in the same session block. Nothing else is touched:
no summarising, no rewording, no chunk removed. On the LongMemEval turn store the
bookkeeping is about a third of the characters the reader receives.

``parse_context_blocks`` inverts ``format_facts`` closely enough that a saved
``full`` context can be re-laid-out offline, so the compact layout can be tested
on frozen contexts with ``rerun-pipeline`` before any retrieval is re-run.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from .models import RetrievedFact

CONTEXT_FORMATS = ("full", "compact")

_HEADER_RE = re.compile(r"^\[(?P<bits>[^\n\]]*)\]$")
_SESSION_RE = re.compile(r"Session (?P<label>\S+) - happened on (?P<date>.+?) UTC\.?")
_UNIT_RE = re.compile(r"^(turn|part) (?P<i>\d+)/(?P<n>\d+)$")
_REPLY_PREFIX = "(in reply to) "
_TRAILER_KEYS = ("date: ", "type: ", "tags: ")


def format_facts(facts) -> str:
    """Original layout: one block per retrieved chunk, chronological, ``---`` separated."""
    if not facts:
        return ""
    sorted_facts = sorted(facts, key=lambda f: (0, f.timestamp) if f.timestamp else (1, ""))
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


@dataclass
class _Chunk:
    body: str
    label: str | None = None
    date_display: str | None = None
    timestamp: str | None = None
    unit_index: int | None = None
    order: int = 0


@dataclass
class _Session:
    label: str
    date_display: str | None
    timestamp: str | None
    first_seen: int
    chunks: list[_Chunk] = field(default_factory=list)


def _parse_stored_header(content: str, order: int, timestamp: str | None) -> _Chunk:
    """Split a stored chunk into its header fields and body."""
    first, _, rest = content.partition("\n")
    m = _HEADER_RE.match(first.strip())
    if not m:
        return _Chunk(body=content.strip(), timestamp=timestamp, order=order)
    label = date_display = None
    unit_index = None
    hdr_ts = None
    for bit in m.group("bits").split(" | "):
        bit = bit.strip()
        if bit.startswith("date: "):
            hdr_ts = bit[len("date: "):].strip()
        elif bit.startswith("context: "):
            sm = _SESSION_RE.search(bit)
            if sm:
                label, date_display = sm.group("label"), sm.group("date")
        else:
            um = _UNIT_RE.match(bit)
            if um:
                unit_index = int(um.group("i"))
    if label is None:
        return _Chunk(body=content.strip(), timestamp=timestamp or hdr_ts, order=order)
    return _Chunk(
        body=rest.strip(),
        label=label,
        date_display=date_display,
        timestamp=timestamp or hdr_ts,
        unit_index=unit_index,
        order=order,
    )


def _strip_reply_prefix(body: str, user_turns_seen: list[str]) -> str:
    """Drop ``(in reply to) <anchor>`` when the anchored user turn is already in the block."""
    if not body.startswith(_REPLY_PREFIX):
        return body
    anchor, sep, remainder = body[len(_REPLY_PREFIX):].partition("\n\n")
    if not sep:
        return body
    anchor = anchor.strip()
    if anchor.endswith("..."):
        stem = anchor[:-3].rstrip()
        present = any(t.startswith(stem) for t in user_turns_seen)
    else:
        present = anchor in user_turns_seen
    return remainder.lstrip() if present else body


def format_facts_compact(facts) -> str:
    """Session-grouped layout with one header per session and no per-chunk bookkeeping."""
    if not facts:
        return ""
    sessions: "OrderedDict[str, _Session]" = OrderedDict()
    loose: list[_Chunk] = []
    for order, f in enumerate(facts):
        ch = _parse_stored_header(f.content or "", order, f.timestamp)
        if ch.label is None:
            loose.append(ch)
            continue
        sess = sessions.get(ch.label)
        if sess is None:
            sess = _Session(label=ch.label, date_display=ch.date_display, timestamp=ch.timestamp, first_seen=order)
            sessions[ch.label] = sess
        sess.chunks.append(ch)

    ordered = sorted(sessions.values(), key=lambda s: ((0, s.timestamp) if s.timestamp else (1, ""), s.first_seen))
    out: list[str] = []
    for sess in ordered:
        chunks = sorted(sess.chunks, key=lambda c: (c.unit_index if c.unit_index is not None else 10**9, c.order))
        when = f", {sess.date_display} UTC" if sess.date_display else ""
        parts = [f"Session {sess.label}{when}"]
        user_turns: list[str] = []
        for ch in chunks:
            body = _strip_reply_prefix(ch.body, user_turns)
            if not body:
                continue
            for para in body.split("\n\n"):
                if para.startswith("User:"):
                    user_turns.append(para.strip())
            parts.append(body)
        out.append("\n\n".join(parts))

    if loose:
        loose.sort(key=lambda c: ((0, c.timestamp) if c.timestamp else (1, ""), c.order))
        out.append("\n\n".join(c.body for c in loose if c.body))
    return "\n\n===\n\n".join(out)


def format_context(facts, style: str = "full") -> str:
    if style == "compact":
        return format_facts_compact(facts)
    if style == "full":
        return format_facts(facts)
    raise ValueError(f"unknown context format {style!r}; expected one of {CONTEXT_FORMATS}")


def parse_context_blocks(context: str) -> list[RetrievedFact]:
    """Recover ``RetrievedFact`` objects from a saved ``full``-layout context."""
    facts: list[RetrievedFact] = []
    if not context:
        return facts
    for block in context.split("\n---\n"):
        lines = block.split("\n")
        timestamp = memory_type = None
        tags: list[str] = []
        while lines and lines[-1].startswith(_TRAILER_KEYS):
            last = lines.pop()
            if last.startswith("date: "):
                timestamp = last[len("date: "):].strip()
            elif last.startswith("type: "):
                memory_type = last[len("type: "):].strip()
            else:
                tags = [t.strip() for t in last[len("tags: "):].split(",") if t.strip()]
        title = None
        hdr_at = next((i for i, ln in enumerate(lines[:2]) if _HEADER_RE.match(ln.strip())), None)
        if hdr_at == 1:
            title = lines[0].strip() or None
            lines = lines[1:]
        content = "\n".join(lines).strip()
        if not content:
            continue
        facts.append(RetrievedFact(id="", content=content, timestamp=timestamp, memory_type=memory_type, title=title, tags=tags))
    return facts


def reformat_context(context: str, style: str) -> str:
    """Re-lay-out a saved ``full`` context in another style (identity for ``full``)."""
    if style == "full":
        return context
    return format_context(parse_context_blocks(context), style)
