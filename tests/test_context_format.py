"""Tests for the reader context layouts (full vs compact)."""

from longmemeval.context_format import (
    format_facts,
    format_facts_compact,
    parse_context_blocks,
    reformat_context,
)
from longmemeval.models import RetrievedFact


def _turn(label: str, date: str, i: int, n: int, body: str, ts: str | None = None) -> RetrievedFact:
    header = f"[date: {ts or date.replace(' ', 'T') + '+00:00'} | context: Session {label} - happened on {date} UTC. | turn {i}/{n}]"
    return RetrievedFact(id=f"{label}-{i}", content=f"{header}\n{body}", timestamp=ts or date.replace(" ", "T") + "+00:00", memory_type="fact", title="Some server title")


USER_A = "User: I adopted a greyhound named Biscuit last weekend, any tips on settling him in?"
LONG_USER = "User: " + "I keep a list of every board game we own and want to rotate them on game night, " * 4 + "which should go first?"


def _facts():
    s1, d1 = "aaaaaaaaaaaa", "2023-03-14 10:00:00"
    s2, d2 = "bbbbbbbbbbbb", "2023-07-02 18:30:00"
    return [
        # arrival order is retrieval order, not chronological and not turn order
        _turn(s2, d2, 2, 2, "User: Did the pottery class fill up?\n\nAssistant: Yes, all eight places went."),
        _turn(s1, d1, 3, 3, f"(in reply to) {LONG_USER[:237].rstrip()}...\n\nStart with the shortest one; rotate weekly."),
        _turn(s1, d1, 1, 3, f"{USER_A}\n\nAssistant: Give him a quiet corner and a routine."),
        _turn(s1, d1, 2, 3, f"{LONG_USER}\n\nAssistant: Here is how I would think about it, in two parts."),
        _turn(s2, d2, 1, 2, "User: I signed up for a pottery class on Tuesdays.\n\nAssistant: Enjoy it."),
    ]


def test_compact_groups_by_session_and_orders_by_turn():
    out = format_facts_compact(_facts())
    # one header per session, chronological
    assert out.count("Session aaaaaaaaaaaa, 2023-03-14 10:00:00 UTC") == 1
    assert out.count("Session bbbbbbbbbbbb, 2023-07-02 18:30:00 UTC") == 1
    assert out.index("Session aaaaaaaaaaaa") < out.index("Session bbbbbbbbbbbb")
    # turns inside a session in turn order
    assert out.index("greyhound") < out.index("board game") < out.index("shortest one")
    assert out.index("signed up for a pottery class") < out.index("Did the pottery class fill up")
    # no per-chunk bookkeeping
    assert "[date:" not in out and "happened on" not in out
    assert "\ndate: " not in out and "\ntype: " not in out and "---" not in out
    assert "Some server title" not in out


def test_compact_drops_reply_anchor_only_when_user_turn_present():
    out = format_facts_compact(_facts())
    # the anchored user turn is in the block, so the truncated repeat is gone but the reply text stays
    assert "(in reply to)" not in out
    assert "Start with the shortest one" in out
    assert out.count("rotate them on game night") == 4  # the original long user turn, once

    # same split piece without its user turn in context: the anchor is kept (it is the only copy)
    piece = _turn("cccccccccccc", "2023-01-01 09:00:00", 2, 2, f"(in reply to) {LONG_USER[:237].rstrip()}...\n\nStart with the shortest one.")
    alone = format_facts_compact([piece])
    assert alone.startswith("Session cccccccccccc, 2023-01-01 09:00:00 UTC")
    assert "(in reply to)" in alone and "Start with the shortest one." in alone


def test_compact_is_lossless_on_conversation_text():
    facts = _facts()
    out = format_facts_compact(facts)
    for f in facts:
        body = f.content.split("\n", 1)[1]
        for para in body.split("\n\n"):
            if para.startswith("(in reply to) "):
                continue
            assert para in out, para[:60]


def test_parse_full_context_round_trips_into_compact():
    facts = _facts()
    full = format_facts(facts)
    assert "Some server title" in full and "type: fact" in full and "\n---\n" in full
    parsed = parse_context_blocks(full)
    assert len(parsed) == len(facts)
    assert {p.title for p in parsed} == {"Some server title"}
    assert all(p.memory_type == "fact" for p in parsed)
    # compact from the saved string equals compact from the live facts
    assert reformat_context(full, "compact") == format_facts_compact(facts)
    assert reformat_context(full, "full") == full


def test_chunks_without_session_header_fall_back_to_a_trailing_block():
    loose = RetrievedFact(id="x", content="A stored note with no header.", timestamp="2023-05-05T00:00:00+00:00")
    out = format_facts_compact(_facts() + [loose])
    assert out.rstrip().endswith("A stored note with no header.")
    assert "\n\n===\n\n" in out
