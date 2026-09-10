# LongMemEval Benchmark for Caura.ai

Long-term memory evaluation harness for **[Caura.ai](https://caura.ai)** on the **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** benchmark ([ICLR 2025](https://arxiv.org/abs/2410.10813)).

## Overview

LongMemEval evaluates chat assistants across five core long-term memory capabilities over long conversational histories:
- **Information Extraction** (`single-session-user`, `single-session-assistant`)
- **Multi-Session Reasoning** (`multi-session`)
- **Knowledge Updates** (`knowledge-update`)
- **Temporal Reasoning** (`temporal-reasoning`)
- **Personal Preferences** (`single-session-preference`)
- **Abstention** (unanswerable questions)

This repository provides an automated, reproducible benchmark environment designed specifically for Caura's memory architecture (including per-question isolation, bulk / extract ingestion modes, reciprocal-rank fusion, and official auto-eval judge integration).

---

## 1. Quick Start

### Installation

Ensure `uv` is installed, then sync the environment:

```bash
uv sync
```

### Environment Configuration

Copy `.env.example` to `.env` (already created with your project credentials if initialized):

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
JUDGE_LLM=gemini
JUDGE_MODEL=gemini-2.5-flash-lite
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

### Run Baselines

Evaluate baseline providers for direct comparison:

```bash
# In-memory keyword/BM25 baseline
uv run longmemeval run --provider bm25 --limit 10 --name bm25-baseline

# Oracle (gold session only) baseline
uv run longmemeval run --provider oracle --limit 10 --name oracle-baseline
```

---

## 4. Evaluating Existing Hypotheses

If you already generated model responses into a JSONL file, you can evaluate them directly with the official judge:

```bash
uv run longmemeval evaluate-hypotheses outputs/caura-run/hypotheses.jsonl --output outputs/caura-run/eval_results.json
```

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
