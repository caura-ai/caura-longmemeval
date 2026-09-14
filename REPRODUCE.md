# Reproducing the Caura LongMemEval_S results

Everything a published number rests on is in this repository: the harness, the
saved reader contexts for every question (`outputs/<run>/hypotheses.jsonl`),
every judge's per-question verdicts (`outputs/<run>/eval_results*.json`),
`results.json` with the effective configuration, the HTML report, and the
scripts that produce the coverage, failure-bucket and token tables.

## Environment

| component | value |
|---|---|
| harness | this repository, commit in the run's `results.json` is the one tagged in the post |
| Python / uv | Python >= 3.11, `uv sync` |
| Caura server | `https://caura.ai/api/v1`, version **3.10.1**, plugin 2.22.1 at the time of the headline run (recorded per run in `results.json -> run.server`; the server reported 3.9.1 / 2.22.0 earlier the same day, before the `lmeo` ingest finished) |
| server embedding | `BAAI/bge-m3` |
| server search | tenant default profile: hybrid (`fts_weight` 0.6, `score_formula` 0, `freshness_reference` 1); `/search` called without `valid_at` |
| server-side derivation LLM | `gemini-3.1-flash-lite` via Vertex (derives its own memories asynchronously; dropped from the reader context in turn mode, see below) |
| dataset | `longmemeval_s_cleaned.json` from `xiaowu0162/longmemeval-cleaned`, 500 questions |
| reader (answering model) | `gemini-3.8-flash`, temperature 0 |
| primary judge | `gpt-4o` (OpenAI alias; resolves to `gpt-4o-2024-08-06`), temperature 0, official task-specific prompts, unmodified |
| secondary judge | `gemini-3.5-flash-lite`, same prompts |

Non-secret `.env` values used by the headline run:

```ini
CAURA_BASE_URL=https://caura.ai/api/v1
READER_LLM=gemini
READER_MODEL=gemini-3.8-flash
JUDGE_LLM=openai
JUDGE_MODEL=gpt-4o
SECONDARY_JUDGE_LLM=gemini
SECONDARY_JUDGE_MODEL=gemini-3.5-flash-lite
```

Every retrieval knob is passed explicitly on the command line below, so the
legacy `CAURA_*` values in `.env` (4k-part store) do not matter for turn mode;
the effective values are also written to `results.json -> run.parameters`.

## Headline run (`outputs/caura-500-opaque`)

Ingest at turn granularity into a fresh store (agent prefix `lmeo`), with
opaque session labels (no dataset session id or question id reaches the stored
text), then run retrieval + generation + both judges:

```bash
uv run longmemeval ingest --name caura-500-opaque-ingest \
    --chunk-mode turns --agent-prefix lmeo --bulk-size 100 --concurrency 5

uv run longmemeval run --name caura-500-opaque --provider caura --skip-ingest \
    --chunk-mode turns --agent-prefix lmeo \
    --sibling-expansion --sibling-window 0 --context-budget 150000 \
    --raw-turns-only --no-as-of-recall \
    --pipeline agentic-v1 --reader gemini --reader-model gemini-3.8-flash \
    --judge openai --judge-model gpt-4o \
    --secondary-judge gemini --secondary-judge-model gemini-3.5-flash-lite \
    --concurrency 5
```

Effective retrieval configuration (turn-mode defaults, all recorded in `results.json`):

| knob | value |
|---|---|
| chunking | one memory per user turn + the assistant reply that follows it, max 1,200 chars; replies longer than that are split and each piece prefixed with `(in reply to) <user turn>` |
| memory header | `[date: <ISO> \| context: Session <opaque 12-hex label> - happened on <date> UTC. \| turn i/n]` |
| search | one `/search` call with the raw question, `top_k` 150 candidates (server default hybrid profile), no `valid_at` |
| server-derived memories | dropped from the candidate list (`--raw-turns-only`, default in turn mode); count recorded per question in `hypotheses.jsonl -> retrieval_stats` |
| expansion | walk candidates in rank order, at most 3 seeds per session, pull the whole session of each seed (`--sibling-window 0`) until the 150,000-character budget is spent |
| context order | chronological (session date, then turn index) |
| category logic | none (`category_adaptive` false); the question type is never read |
| reader pipeline | `agentic-v1`: extract evidence from the full context, draft, conditional infer / direct fallback, verify. Exact per-call token usage recorded in `hypotheses.jsonl -> reader_usage` |

## Judging an existing run with another judge

```bash
uv run longmemeval evaluate-hypotheses outputs/caura-500-opaque/hypotheses.jsonl \
    --judge openai --judge-model gpt-5.6-terra \
    -o outputs/caura-500-opaque/eval_results_gpt56terra.json --no-html
```

## Coverage, failure buckets, tokens

```bash
# gold-turn coverage and retrieval-/reader-bound buckets under each judge file present
uv run python scripts/coverage.py outputs/caura-500-opaque --list-failures --json outputs/caura-500-opaque/coverage.json

# context tokens (Gemini tokenizer), total reader tokens per question, latency
uv run python scripts/context_tokens.py outputs/caura-500-opaque
```

## Runs in this repository

| run | date | store | what it is | GPT-4o | Flash-Lite |
|---|---|---|---|---|---|
| `caura-500-agentic-v1` | 10 Sep 2026 | 4,000-char session parts, category-keyed retrieval profiles | baseline; Flash-Lite judged 10 Sep, GPT-4o / GPT-5.6 judged 13 Sep | 88.8 | 87.0 |
| `caura-500-turns` | 12 Sep 2026 | turn granularity (`lmet`), headers carry dataset session ids, derived memories allowed | first turn-granularity run; the 91.4% draft number | 91.4 | 89.0 |
| `caura-500-turns-0914` | 14 Sep 2026 | same `lmet` store, fresh retrieval + generation | variance run of the above (same configuration) | 91.0 | 89.4 |
| `caura-500-turns-prompts-v2` | 14 Sep 2026 | saved `caura-500-turns` contexts | reader prompts with dataset-lifted examples removed, re-run on frozen contexts (`rerun-pipeline`) | 90.8 | 89.8 |
| `caura-500-opaque` | 14 Sep 2026 | turn granularity (`lmeo`), opaque labels, derived memories dropped, new prompts | **headline**: all three review fixes applied | **91.6** | 89.8 |
| `caura-500-opaque-run2` | 14 Sep 2026 | same `lmeo` store, fresh retrieval + generation | variance run of the headline (identical configuration) | see `eval_results.json` | |
| `oracle-500-agentic-v1` | 14 Sep 2026 | no store: reader gets exactly the answer sessions | reader ceiling under perfect retrieval (`--provider oracle`) | 94.6 | 92.8 |

Scores are accuracy over all 500 questions from each run's `eval_results.json`
(GPT-4o) and `eval_results_gemini35flashlite.json`. `caura-500-turns` and
`caura-500-agentic-v1` also carry `eval_results_gpt56terra.json` and
`eval_results_gpt56luna.json` from the 13 September four-judge comparison.

Baselines:

```bash
# oracle: the reader sees exactly the question's answer sessions (abstention questions get their related sessions)
uv run longmemeval run --provider oracle --name oracle-500-agentic-v1 --pipeline agentic-v1 \
    --reader gemini --reader-model gemini-3.8-flash --judge openai --judge-model gpt-4o \
    --secondary-judge gemini --secondary-judge-model gemini-3.5-flash-lite --concurrency 5

# full context: the reader sees the whole haystack (~115k tokens per question; not run yet, ~75M reader input tokens)
uv run longmemeval run --provider fullcontext --name fullcontext-500-agentic-v1 --pipeline agentic-v1 \
    --reader gemini --reader-model gemini-3.8-flash --concurrency 3
```

Other full-500 evaluations made during development, not committed because they
are single-judge runs on the superseded 4k store: 11 Sep `caura-500-as-of-v4`
(`valid_at` on, Flash-Lite 428/500) and `caura-500-restore` (`valid_at` off,
434/500); the 11 Sep map-reduce-by-default run scored 81.2% under Flash-Lite and
was reverted (commit d69343c).

The judge alias is pinned in the artifacts: `eval_results*.json -> judge_model_resolved`
carries the dated snapshot the API served (for `gpt-4o`, `gpt-4o-2024-08-06`).

## Cleaning up

```bash
uv run python scripts/caura_cleanup.py            # dry run
uv run python scripts/caura_cleanup.py --apply    # delete lme-* agents and their memories
```
