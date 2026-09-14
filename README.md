# Caura on LongMemEval: 91.6%

**[Caura](https://caura.ai)'s memory API answers 458 of 500 LongMemEval_S questions correctly**, scored with the benchmark's reference judge (GPT-4o, official prompts, unmodified), with Gemini 3.8 Flash as the answering model and one configuration for all six question types.

| | GPT-4o judge | Gemini 3.5 Flash-Lite judge (stricter) |
|---|---|---|
| **Overall** | **91.6%** (458/500) | 89.8% (449/500) |
| single-session user | 94.3% | 94.3% |
| single-session assistant | 100.0% | 98.2% |
| multi-session | 85.7% | 80.5% |
| temporal reasoning | 90.2% | 91.0% |
| knowledge update | 97.4% | 92.3% |
| preference | 86.7% | 93.3% |

Every published LongMemEval number is a joint property of the memory system, the answering model and the judge, so here is how this one was made:

- **No fact extraction, no rewriting.** Caura stores the conversation turns as written and retrieves them with its default hybrid (dense + full-text) search. The answering model sees stored conversation, not LLM-generated summaries.
- **No labels, no shortcuts.** Session ids and question types from the dataset never reach the store or the reader. All 500 questions, abstention items included. No cherry-picked subsets.
- **A Flash-class answering model.** Most published numbers above 90 use Pro- or Opus-class answerers. Ours is Gemini 3.8 Flash; with the same pipeline handed perfect retrieval (oracle), the ceiling is 94.6%, so retrieval costs about three points and the rest is the reader.
- **Two judges, always both.** GPT-4o is the headline because it is the benchmark's reference judge; Gemini 3.5 Flash-Lite, which fails hedged answers, is reported beside it on every run. Judge snapshots are recorded in every verdict file.
- **Measured, not estimated.** Median context handed to the reader: 48.6k tokens (Gemini tokenizer, 32% of the haystack); total reader tokens per question across all four pipeline stages: 52.7k median. A second identical run scored 92.0% / 89.2%.

**Everything behind the number is in this repository.** The saved reader context for every question, every judge's per-question verdicts, `results.json` with the effective configuration, the HTML reports, and the scripts that produce the coverage and token tables are committed under `outputs/`. Exact commands, environment and Caura server version: [`REPRODUCE.md`](REPRODUCE.md).

## How it works

LongMemEval ([Wu et al., ICLR 2025](https://arxiv.org/abs/2410.10813)) gives each of 500 questions a haystack of ~50 chat sessions (~115k tokens) and asks something the user said months earlier. Question types: single-session user, single-session assistant, multi-session, temporal reasoning, knowledge update, preference, plus 30 unanswerable items where the right answer is to say so.

The configuration behind 91.6% is **store the turn, read the session**:

1. **Ingest at turn granularity.** Each user turn plus the assistant reply that follows becomes its own Caura memory (~800 characters median), written through the standard bulk endpoint with the session date in a header. Small units embed cleanly: the sentence that carries the answer ranks on its own instead of being averaged into a 4,000-character chunk about three other topics.
2. **Read at session granularity.** One hybrid search with the raw question, then pull the rest of each hit's session from the store, in chronological order, under a fixed 150k-character budget. The turn finds the right session; the whole session supplies the number or date the question needs.
3. **A four-stage reader** on Gemini 3.8 Flash: extract evidence, draft, infer or fall back, verify against the evidence.

Gold-turn coverage, the share of answer-carrying turns that reach the reader, is 0.974 with this configuration. Of the 42 remaining errors under GPT-4o, 17 are retrieval misses and 22 are the reader getting it wrong with the evidence in front of it; `scripts/coverage.py` reproduces that split from the committed outputs.

This repository is the harness that produced the number: dataset download, ingestion into Caura at either granularity, retrieval with session expansion, the reader pipeline, both judges with the official prompts, oracle / full-context / BM25 baselines, HTML reports, and the coverage and token-accounting scripts.

---

## 1. Quick Start

### Installation

Ensure `uv` is installed, then sync the environment:

```bash
uv sync
```

### Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```ini
# Caura API credentials
CAURA_API_KEY=mc_...
CAURA_TENANT_ID=ten2-...
CAURA_BASE_URL=https://caura.ai/api/v1
CAURA_AGENT_PREFIX=lme

# Ingestion configuration
CAURA_INGEST=bulk          # "bulk" or "extract"
CAURA_CHUNK_CHARS=4000
CAURA_BULK_SIZE=25
CAURA_TOP_K=20

# Evaluation & Answer Generation LLM
GEMINI_API_KEY=AIzaSy...
READER_LLM=gemini
READER_MODEL=gemini-3.8-flash

# Judge protocol: gpt-4o primary (headline), gemini-3.5-flash-lite secondary; both always reported
OPENAI_API_KEY=sk-...
JUDGE_LLM=openai
JUDGE_MODEL=gpt-4o
SECONDARY_JUDGE_LLM=gemini
SECONDARY_JUDGE_MODEL=gemini-3.5-flash-lite
```

---

## 2. Dataset Setup

Download the official cleaned dataset (`longmemeval_s_cleaned.json`, ~115k tokens per haystack, 500 questions):

```bash
uv run longmemeval download
```

You can view category breakdown and dataset stats:

```bash
uv run longmemeval stats
```

---

## 3. Running Benchmarks

### Smoke Test (1-2 questions)

Run a fast smoke test against Caura:

```bash
uv run longmemeval run --provider caura --limit 2 --name smoke-test
```

### Full Benchmark or Specific Category

Run on a specific LongMemEval category:

```bash
# Temporal reasoning questions
uv run longmemeval run --provider caura --category temporal-reasoning --name caura-temporal

# Knowledge update questions
uv run longmemeval run --provider caura --category knowledge-update --name caura-update
```

Available categories:
- `single-session-user`
- `single-session-assistant`
- `multi-session`
- `temporal-reasoning`
- `knowledge-update`
- `single-session-preference`

### Turn-granularity store (the published configuration)

Ingest once into a dedicated store, then run against it. Session headers carry an opaque label
(never the dataset's `answer_*` session id or the question id), server-derived memories are dropped
from the search results before session expansion (`--raw-turns-only`, default in turn mode), and the
exact reader token usage per question is recorded in `hypotheses.jsonl`:

```bash
uv run longmemeval ingest --name my-turn-ingest --chunk-mode turns --agent-prefix lmeo --bulk-size 100
uv run longmemeval run --name my-turn-run --skip-ingest --chunk-mode turns --agent-prefix lmeo \
    --sibling-expansion --context-budget 150000 --raw-turns-only --no-as-of-recall --pipeline agentic-v1
```

### Run Baselines

Evaluate baseline providers for direct comparison:

```bash
# In-memory keyword/BM25 baseline
uv run longmemeval run --provider bm25 --limit 10 --name bm25-baseline

# Oracle: the reader sees exactly the answer sessions (reader ceiling under perfect retrieval)
uv run longmemeval run --provider oracle --pipeline agentic-v1 --name oracle-baseline

# Full context: the reader sees the whole haystack, no retrieval
uv run longmemeval run --provider fullcontext --pipeline agentic-v1 --name fullcontext-baseline
```

### Coverage, failure buckets, token budget

```bash
# gold-turn coverage; every failure bucketed as retrieval-bound (gold turn missing) or reader-bound
uv run python scripts/coverage.py outputs/my-turn-run --list-failures

# context tokens with the reader's tokenizer, total reader tokens per question, latency medians
uv run python scripts/context_tokens.py outputs/my-turn-run
```

---

## 4. Evaluating Existing Hypotheses

If you already generated model responses into a JSONL file, you can evaluate them directly with the official judge prompts. By default both protocol judges run and are written next to the hypotheses (`eval_results.json` for the primary, `eval_results_gemini35flashlite.json` for the secondary), and `results.json` gets a `secondary_evaluation` block with per-category numbers and agreement:

```bash
uv run longmemeval evaluate-hypotheses outputs/caura-run/hypotheses.jsonl
```

For an ad-hoc single-judge comparison, name the judge and an output file; only that judge runs:

```bash
uv run longmemeval evaluate-hypotheses outputs/caura-run/hypotheses.jsonl --judge openai --judge-model gpt-5.6-terra -o outputs/caura-run/eval_results_gpt56terra.json --no-html
```

### Judge protocol

Fixed on 13 September 2026 after re-judging two full-500 runs with four judges (Flash-Lite had been the only judge from 10 to 13 September):

- **Primary: `gpt-4o`.** The LongMemEval reference judge and the one most published leaderboard numbers use. Its score is the headline and goes in `eval_results.json`; the dated snapshot the API served (`gpt-4o-2024-08-06`) is recorded as `judge_model_resolved`.
- **Secondary: `gemini-3.5-flash-lite`.** Stricter (fails hedged answers such as "2 or 3"), cheap, and useful for development because it penalises exactly the reader behaviour we want to remove. Always reported beside the primary, never instead of it.
- Judging costs under $1 per run per model, so neither is dropped. The judge is never chosen after seeing the score; `--judge`/`--secondary-judge` exist for controlled judge studies, and their outputs are labelled with the model name.

On the 89.0% (Flash-Lite) turn-granularity run the four judges scored: gpt-4o 457/500 (91.4%), gpt-5.6-terra 457, gpt-5.6-luna 461, gemini-3.5-flash-lite 445. Pairwise agreement among the OpenAI judges was 488–492/500; Flash-Lite agreed with each on 480–482.

---

## 5. Cleaning Up Caura Tenant Data

All memories written during the benchmark are tagged with `CAURA_AGENT_PREFIX` (default: `lme-*`). To clean up:

```bash
# Dry run (inspect agents and memories created)
uv run python scripts/caura_cleanup.py

# Apply deletion
uv run python scripts/caura_cleanup.py --apply
```

---

## 6. Architecture & Isolation Details

1. **Per-Question Isolation**:
   - Each question has its own distinct haystack of conversation sessions.
   - The provider maps `question_id` to both a dedicated Caura `agent_id` (`lme-{question_id}`) and `fleet_id` (`lme-{question_id}`).
   - Retrieval queries pass `filter_agent_id`, preventing cross-question memory leakage and ensuring pure needle-in-haystack evaluation.

2. **Ingestion Modes**:
   - `bulk`: Sends document conversation chunks (up to 4,000 characters) to `POST /memories/bulk`. Retains session timestamp and context header metadata.
   - `extract`: Sends session dialogues to `POST /ingest/preview` and commits extracted atomic facts via `POST /ingest/commit`.

3. **Official Judge Compatibility**:
   - The judge prompts in `longmemeval.prompts` exactly match the official ICLR 2025 LongMemEval prompt templates, including task-specific rules (e.g. off-by-one day tolerance for `temporal-reasoning`, update tolerance for `knowledge-update`, rubric matching for `single-session-preference`).
