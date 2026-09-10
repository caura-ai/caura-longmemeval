# How to Excel on Agent Memory Benchmarks

Engineering report · 10 September 2026

> **One week of progress:** Over the last seven days, Caura moved from scattered, partly invalid benchmark runs to a credible measurement program across four complementary suites. We established a full 85.4% PersonaMem result, repaired LongMemEval's missing-history baseline and reached 171/200 (85.5% research-optimized; 82.4% projected on the official mix), and lifted the observed BEAM 100K pass rate from 41.0% to 47.0% in the newer harness while discovering that `top_20` beats `top_200` at one fifth of the context. Most recently, an agentic reader crossed 80% on a question-disjoint LoCoMo holdout: **33/40 answerable questions, or 82.5%**, with 99.2% annotated-evidence recall. The paired direct reader scored 77.5% on the same Caura memories, proving the five-point lift belongs to answer orchestration rather than the memory server; the agentic path also reduced adversarial abstention from 100% to 50%. Just as importantly, we exposed the causes behind misleading numbers—contaminated scopes, incomplete ingestion, gold-label adaptation, judge variance, and protocol/model mismatches. The result is more than a higher score: we now know which gains belong to Caura, which belong to the harness, and which server improvements should produce the next durable step-change.

## Executive summary

Agent-memory benchmark scores are not a direct measurement of the memory server. They are the output of a pipeline:

`dataset → ingestion → storage → retrieval → context packing → answer model → scorer or judge`

A change at any layer can move the headline score. The best teams therefore do two things at once:

1. improve the memory architecture; and
2. make the evaluation protocol explicit, reproducible, and hard to accidentally game.

Caura's local evidence is encouraging but uneven:

- **AMB / PersonaMem 32k:** 503/589, **85.4%**, using raw 2k-character chunks, `top_k=50`, Gemini 3.8 Flash as answerer, and exact MCQ-letter scoring. The output records a Gemini 2.5 Flash-Lite judge configuration, but PersonaMem does not invoke that judge. This is our strongest full, non-label-leaking result. The local summary places it 1.0 point above Hybrid Search and 1.2 points below Hindsight, with materially faster ingestion; the raw comparator artifacts are not present in this checkout, so those cross-provider deltas should be treated as reported rather than independently reverified here.
- **LongMemEval:** corrected 171/200, **85.5%**, using Gemini 3.8 Flash and Gemini 3.5 Flash-Lite. Because the harness directly uses the benchmark's `question_type` label to choose retrieval settings, this is a research-optimized result, not a fair production-style headline. Reweighting its category rates to the official 500-question mix projects roughly **82.4%**.
- **BEAM 100K:** **47.0% pass rate** and **0.450 average nugget score** at `top_200`; `top_20` was better at **49.0% / 0.492** while using one fifth of the answer context. Summarization remained 0%, and event ordering 5%, exposing architectural gaps that larger `top_k` does not solve.
- **LoCoMo:** the newest balanced, question-disjoint holdout scored **82.5% semantic accuracy (33/40 answerable questions; 95% CI 68.0–91.3%)**, with **100% evidence hit rate** and **99.2% annotated-evidence recall**. It used 4k-character production bulk memories, `top_k=30`, one literal query, and Gemini 3.8 Flash as both reader and judge. The ten adversarial questions are reported separately and scored 50% abstention. This is our strongest current LoCoMo semantic result, but it is a 50-question balanced holdout—not the full 1,540-question modern subset or the original token-F1 protocol.

The highest-value work is:

1. freeze comparable and research-optimized protocols as separate tracks;
2. make `question_date → valid_at → temporal scoring` correct end to end;
3. add coverage-aware retrieval for aggregation, summarization, and event ordering;
4. maintain explicit current and historical preference/constraint state;
5. use multiple memory granularities: raw evidence, atomic facts, session summaries, and longitudinal state;
6. measure retrieval, reader, and judge quality separately.

## 1. What each benchmark measures

### LoCoMo

LoCoMo contains ten long, multi-session conversations, averaging roughly 600 utterances and 16K tokens, across up to 32 sessions. It includes question answering, event summarization, and multimodal dialogue generation. The ACL evaluation release is a ten-conversation subset of an earlier 50-conversation release, so dataset version must be stated.

Its QA categories are:

- category 1: multi-hop;
- category 2: temporal;
- category 3: open-domain or commonsense;
- category 4: single-hop;
- category 5: adversarial or unanswerable.

The original paper reports answer F1. Most modern vendor reports instead use a binary LLM "J-score" and exclude category 5, usually evaluating 1,540 questions from categories 1–4. Those numbers are not directly comparable to the original paper's F1 or to runs using a different judge prompt.

LoCoMo is now small enough for strong long-context models to read almost directly. It remains useful for preference evolution, temporal QA, and retrieval diagnostics, but is weak evidence for production-scale memory unless token cost and latency are reported beside accuracy.

Primary sources: [official repository](https://github.com/snap-research/locomo) and [ACL 2024 project page](https://snap-research.github.io/locomo/).

### LongMemEval

LongMemEval contains 500 evaluation instances across six types:

- single-session user facts;
- single-session assistant facts;
- single-session preference;
- temporal reasoning;
- knowledge update;
- multi-session reasoning.

Question IDs ending in `_abs` are abstention cases. Each item includes a `question_date`, timestamped haystack sessions, and evidence annotations. LongMemEval_S averages about 115K tokens and roughly 40 history sessions; LongMemEval_M is much larger, at roughly 500 sessions per item.

The official evaluation uses a task-specific LLM judge. The reference implementation uses GPT-4o for judging and explicitly allows leniency for temporal off-by-one errors and some knowledge-update responses. Reader method also matters: the authors recommend a "chain of note" style that extracts useful facts before reasoning.

LongMemEval is a joint test of:

- whether the right evidence is stored;
- whether it can be retrieved;
- whether dates survive ingestion;
- whether multiple sessions are combined correctly; and
- whether the reader can reason over a large, noisy context.

Primary sources: [official repository](https://github.com/xiaowu0162/LongMemEval) and [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf).

### AMB

AMB is an evaluation harness and leaderboard, not one dataset. It runs a common sequence—ingest, retrieve, answer, judge—over datasets including PersonaMem, LoCoMo, LongMemEval, BEAM, and coding or research tasks.

Our main AMB work uses PersonaMem 32k:

- 37 personas;
- 195 documents;
- 589 multiple-choice queries;
- seven question types covering fact recall, preference evolution, reasons for updates, generalization, recommendations, and new suggestions.

AMB's main contribution is protocol transparency: outputs record the answer and judge models, context, timing, and per-query decisions. It also supports oracle mode, which is essential for separating retrieval failures from answer-model failures.

For PersonaMem specifically, scoring is deterministic option-letter equality rather than an LLM judge. Its retrieval query also excludes the MCQ options while the answerer sees the full question. This avoids using answer choices as lexical retrieval hints and should remain frozen in comparisons.

Primary sources: [official repository](https://github.com/vectorize-io/agent-memory-benchmark), [leaderboard](https://agentmemorybenchmark.ai/), and [PersonaMem](https://github.com/bowen-upenn/PersonaMem).

### BEAM

BEAM targets scale and breadth:

- 100 coherent conversations;
- 2,000 human-validated questions;
- 128K, 500K, 1M, and 10M-token settings; AMB and Mem0 commonly label the smallest split "100K";
- ten memory abilities.

The abilities are abstention, contradiction resolution, event ordering, information extraction, instruction following, knowledge update, multi-session reasoning, preference following, summarization, and temporal reasoning.

BEAM judges fine-grained rubric nuggets on a 0, 0.5, or 1 scale. Event ordering also uses Kendall's tau-b. A harness may report both:

- **average nugget score**, and
- **pass rate**, where a question passes when its average nugget score is at least 0.5.

These must not be conflated. For example, Mem0 reports BEAM 1M at `top_200` as 70.1% pass rate and 0.641 average score. A headline of "64.1" refers to the score expressed as a percentage, not the pass rate.

We have two local BEAM paths. The dedicated `yanki-beam` harness evaluates several retrieval cutoffs and can calculate event-ordering tau. The AMB BEAM adapter uses a different judge/scoring path and a single retrieval pass. It deliberately applies nugget compliance to event ordering for Mem0 head-to-head parity instead of the paper's Kendall tau-b metric. Results from the two paths should not be merged.

Version also matters: the official repository's 31 August 2026 revision records an evaluation-scoring bug fix. Pin the dataset and evaluator commit for every BEAM claim.

Primary sources: [official repository](https://github.com/mohammadtavakoli78/BEAM) and [ICLR 2026 paper](https://arxiv.org/abs/2510.27246).

## 2. The benchmark, harness, and Caura are different systems

### Benchmark layer

The benchmark controls:

- dataset version and ground truth;
- category definitions and category mix;
- full, sampled, or oracle evaluation;
- answer and scoring task;
- whether adversarial or abstention questions count.

Changing this layer usually invalidates comparison.

Use an ingestion allowlist, not a denylist. Only conversational content and production-available metadata may enter memory. LongMemEval's `has_answer` and `answer_session_ids`, LoCoMo QA answers/evidence, and BEAM rubrics are gold data and must remain evaluator-side.

### Harness layer

The harness controls:

- how conversations become documents or memories;
- chunk size and overlap;
- whether speaker, session, and date tags are preserved;
- query rewriting and repeated searches;
- `top_k`, candidate merging, and context truncation;
- sort order;
- answer prompt and answer model;
- judge prompt, judge model, and parse-failure behavior;
- retries, resumption, caching, and handling of failures.

The harness can improve the score without improving the memory server. It can also suppress a good server with poor packing or a weak reader.

### Caura server layer

The server controls:

- enrichment and atomic fact extraction;
- embeddings, lexical search, entities, and graph expansion;
- temporal validity and contradiction state;
- deduplication and supersession;
- candidate generation, scoring, and reranking;
- identity and fleet scoping;
- latency, rate limits, diagnostics, and native grounded recall.

The server should receive only information available in production. Passing a benchmark's gold category label directly to retrieval is not production-realistic.

## 3. Parameters that materially affect results

### Dataset and sampling

Always pin:

- dataset URL, version, and content hash;
- split and scale;
- exact question IDs;
- category inclusion and weighting;
- random seed and ordering;
- whether the run is full, sampled, balanced, or oracle.

A category-balanced sample can overstate or understate the official weighted score. Our LongMemEval 85.5% becomes about 82.4% when projected onto the official category mix.

### Ingestion granularity

Important settings include:

- turn, session, document, or fixed-size chunk;
- characters versus tokens;
- overlap and boundary rules;
- raw bulk storage versus LLM extraction;
- extraction model, prompt, and output limit;
- fact salience threshold;
- speaker, session, source, and timestamp metadata;
- deduplication and overwrite/supersession policy.

Our AMB result demonstrates the interaction:

- 8k raw chunks at `top_k=20`: 82.9%;
- 2k raw chunks at `top_k=20`: not enough coverage;
- 2k raw chunks at `top_k=50`: 85.4%;
- extracted facts at `top_k=50`: 71.6%, but only 2.3K context tokens.

There is no globally best chunk size. Smaller chunks improve selectivity only if `top_k` grows enough to preserve coverage.

### Answer or reader model

Pin provider, exact model version, temperature, maximum output, system prompt, and reasoning mode.

Reader strength can move results by several points on identical memory context. In our 50-question AMB model screen:

- Gemini 3.8 Flash: 84%;
- Claude Sonnet 4.6: 84%;
- Grok 4.3: 78%;
- Gemini 3.1 Pro Preview: 76%.

The larger or more expensive model was not automatically better. The answer format and task fit matter.

In LongMemEval, six analyzed adaptive-v4 failures already contained every gold session in the reader context. Those were reasoning and context-use failures, not evidence-retrieval failures.

### Judge model

Pin judge provider, exact model, prompt, temperature, response schema, retry policy, and parse-failure behavior.

Rejudging the same 54 LongMemEval answers produced:

- Gemini 3.5 Flash-Lite: 47/54, 87.0%;
- Gemini 3.8 Flash: 47/54, 87.0%, with different per-question decisions;
- GPT-4o: 45/54, 83.3%;
- Grok 4.6: 44/54, 81.5%.

The 5.5-point span changed no stored memory, retrieval, context, or answer. Never select the judge after seeing which one gives the highest score. For development, preserve one canonical judge and use a second judge or human sample to estimate uncertainty.

PersonaMem is the exception in our AMB work: its score is exact MCQ-letter match, so the configured `judge_llm` field does not affect the result.

### `top_k`, candidate pool, and context budget

`top_k` is both a recall budget and a noise budget. It interacts with chunk size, reranking, and reader context length.

Our evidence:

- AMB improved when 2k chunks moved from insufficient coverage to `top_k=50`.
- AMB's 2k `top_k=65` plus altered FTS weighting scored 83.7%, below the 2k `top_k=50` run at 85.4%.
- BEAM 100K scored 49.0% pass / 0.492 average at `top_20`, versus 47.0% / 0.450 at `top_200`.
- BEAM answer context grew from about 23.4K to 120.9K tokens between those cutoffs.
- LongMemEval's larger contexts often already contained the evidence; more context did not guarantee a correct answer.

Report both retrieval depth and actual reader prompt tokens. "Top 200" can mean 200 tiny facts or 200 multi-kilobyte chunks.

### Sorting and packing

Common policies are:

- relevance order;
- chronological oldest-to-newest;
- newest-to-oldest;
- relevance-ranked sessions with chronological order inside each session;
- one or more results per source/session before filling remaining slots.

Global chronological sorting can destroy the ranking signal. Pure relevance order can make temporal reconstruction difficult. A strong default is:

1. rank sessions or sources by relevance;
2. enforce source/session diversity;
3. preserve chronology inside each selected session;
4. put dates and status beside every memory;
5. cap total tokens, not only item count.

Our LongMemEval harness currently sorts every fact oldest-to-newest after retrieval. Our BEAM harness does the same. Both should be A/B tested against grouped relevance order under a fixed context budget.

### Date tagging and temporal reference

Date handling needs four distinct fields:

- ingestion time;
- event or valid-start time;
- valid-end time;
- query reference time.

The answer prompt needs the query date. Search also needs that date as `valid_at`.

Our LongMemEval runner passes `question_date` into the provider, but the Caura provider does not forward it to `/search`. The server can already accept `valid_at`, so this is currently lost at the adapter boundary.

The server's first-stage freshness calculation also uses wall-clock `now()` and an anchor based on the greater of `created_at` and `ts_valid_start`. When historical benchmark events are ingested today, `created_at` dominates and makes them all look equally fresh. Temporal windows also use `created_at`. For historical evaluation—and for imported production archives—freshness should be calculated from:

- reference time: `valid_at`, falling back to now;
- event time: `ts_valid_start`, falling back to `created_at`.

`created_at` should not override a known event date.

### Retrieval configuration

Pin and disclose:

- embedding model and query instruction;
- dense/lexical weight;
- full-text rank scaling;
- minimum similarity;
- candidate pool size;
- score formula;
- reranker model and rerank pool;
- entity retrieval and graph-hop limits;
- query rewriting and number of searches;
- merge method and RRF constant;
- recency, status, recall-count, and temporal boosts.

Our AMB 257-question paired sweep found:

- default 8k baseline: 80.9%;
- pure vector (`fts_weight=0`): 80.5%;
- `candidate_pool_size=100`: 81.3%;
- unified `score_formula=1`: 82.1%;
- `fts_weight=0.6`: 82.9%;
- three-query fusion with 40 merged slots: 82.9%.

On the full 589-question Pro run, `fts_weight=0.6` improved 83.0% to 84.0%. These are promising signals, not proof of a universal production default.

### Execution and integrity

Pin:

- Caura server release or Git SHA;
- tenant search profile and feature flags;
- API/harness commit;
- run ID and agent/fleet naming;
- concurrency and rate limits;
- retry and timeout policy;
- settle time;
- resume behavior;
- cache state;
- failure accounting.

Failures must count as failures. A resume must not silently reuse another configuration's hypotheses or memories.

Our corrected LongMemEval baseline illustrates the risk: a `--skip-ingest` run was missing five assistant-history stores. Repairing those five with the same reader, judge, and retrieval settings raised the score from 167/200 to 171/200.

Our AMB work also found a contaminated extract run in which old bulk chunks survived under reused persona scopes. Run-versioned scope IDs and a preflight memory-count check would have prevented it.

Two current failure paths also need explicit counters: the LongMemEval extract provider catches per-document ingestion exceptions and continues, while AMB marks empty retrieval as incorrect without calling the judge. Both are defensible execution choices, but the report must distinguish ingestion, retrieval, answer, and judge failures.

## 4. What competitors are actually tuning

Public evidence mostly shows **harness and system configuration tuning**, not hidden fine-tuning of model weights. The distinction matters.

### Mem0

Mem0's April 2026 benchmark stack discloses:

- single-pass retrieval;
- `top_k=200`, with top-50 ablations;
- GPT-4o answerer and judge defaults in the open harness;
- hierarchical fact extraction;
- semantic, keyword, and entity retrieval fused into one ranking;
- temporal classification at write time;
- roughly 6.9K answer-context tokens.

Its headline results are for the managed platform, which includes proprietary optimizations not present in the open-source SDK. Mem0 also publishes LongMemEval extraction-model ablations while keeping the embedder, vector store, answerer, and judge fixed. That is a good example of isolating one stage.

Mem0's repository history also explicitly records tuning LoCoMo answerer and judge prompts. Prompt tuning is legitimate when disclosed and frozen before comparisons; it makes old and new result sets different protocols.

The current prompt rules are materially benchmark-specific: they constrain LoCoMo dates to 2022–2024, add temporal and counting heuristics, and direct the reader not to abstain on the scored category subset. These are disclosed test-distribution adaptations, not evidence that the same default system will generalize unchanged.

Reproducibility requires the actual artifact and dependency versions. Mem0 states that its managed v3 results include proprietary features unavailable in the OSS SDK, and maintainers have confirmed that the 2025 paper used the SaaS client rather than the OSS `Memory` implementation. An unresolved July 2026 harness report also alleges that an SDK rename caused nominal top-50/top-200 OSS runs to use the default top-20; affected runs should be audited before reuse.

Sources: [benchmark repository](https://github.com/mem0ai/memory-benchmarks) and [research description](https://mem0.ai/research).

### Zep / Graphiti

Zep's LongMemEval work combines a temporal knowledge graph with BGE-m3 embeddings and reranking. It reports both GPT-4o-mini and GPT-4o response-generation configurations and uses GPT-4o with LongMemEval's question-specific evaluation prompts.

This is benchmark-specific configuration, but it maps to a production capability: entity and relationship history with temporal validity. Direct comparison still requires matching answer and judge models.

Its published LoCoMo path retrieves up to 20 nodes with reciprocal-rank fusion plus 20 edges with cross-encoder reranking, uses GPT-4o-mini as reader and judge, and excludes adversarial category 5. Zep later swept node/edge budgets and found gains flattening near 20/20. Its corrected official result is 75.14% ±0.17 across ten runs, under that modern reduced-category J-score protocol—not original LoCoMo F1.

Source: [Zep temporal knowledge graph paper](https://arxiv.org/abs/2501.13956).

### Hindsight and AMB

Hindsight separates retain, recall, and reflect, and stores world facts, experiences, entity summaries, and beliefs. Its paper fixes a GPT-OSS-120B judge at temperature zero while varying the underlying memory/backbone configuration.

Hindsight also budgets retrieved context by tokens—12,288 for BEAM and 32,768 elsewhere—rather than by memory count. Comparing its budget to another system's `top_k` without reporting actual context tokens is not meaningful.

AMB also exposes RAG, agentic-RAG, native direct-answer, and oracle modes. These are different tasks:

- one-shot RAG measures provider retrieval plus a shared reader;
- agentic-RAG measures iterative search policy plus memory;
- native answer measures the provider's full stack;
- oracle isolates reader quality.

A system should only be compared within the same mode.

Sources: [Hindsight paper](https://arxiv.org/abs/2512.12818) and [AMB repository](https://github.com/vectorize-io/agent-memory-benchmark).

### Letta

Letta's LoCoMo experiment stores conversation sessions as files and lets a GPT-4o-mini agent repeatedly call `search_files` until it decides to answer. Tool rules require search first, but query choice and search count are agentic. Letta reports 74.0%; the historical post does not pin the judge version, while the current unmaintained repository defaults elsewhere to GPT-4.1.

That setup can be useful product evidence, but it is not directly comparable to single-call `top_k` retrieval. It measures the agent's search policy and tool use as well as memory.

Sources: [LoCoMo implementation](https://github.com/letta-ai/letta-leaderboard/blob/main/leaderboard/locomo/locomo_benchmark.py) and [Letta's analysis](https://www.letta.com/blog/benchmarking-ai-agent-memory/).

### LangMem and OpenAI

No primary official LangMem or OpenAI submission was found for the four benchmarks. The public LangMem LoCoMo result is Mem0's two-agent implementation and excludes adversarial questions; label it as such. Mem0's ChatGPT Memory experiment supplied all generated memories as context, and LongMemEval's ChatGPT test covered only 97 short-history questions. Neither is a full, selective-retrieval benchmark for the vendor's production memory.

### Common comparability adjustments

Across published reports, the largest score-moving choices are:

- excluding LoCoMo adversarial questions;
- replacing original token F1 with a lenient LLM J-score;
- using stronger answer models;
- changing judge models or prompts;
- increasing `top_k` to 90 or 200;
- adding category-specific answer instructions;
- using repeated agentic retrieval instead of one search;
- using managed proprietary features while publishing an open harness;
- selecting a favorable context budget or benchmark split.

None is automatically improper. The problem is presenting results as directly comparable when the protocol differs.

Direct use of a gold benchmark label—such as LongMemEval `question_type`—is a stronger issue. It is label leakage unless that label is also available in the intended product environment. A production query classifier may infer the same intent from query text, but the benchmark label itself should not be passed to retrieval in a comparable run.

Direct ingestion of answers, evidence-session IDs, `has_answer`, or grading rubrics is stronger still: it contaminates memory with the target. Add automated schema allowlists and post-ingest scans so these fields cannot reach Caura.

## 5. Lessons from our trials

### AMB / PersonaMem

Current best:

- output: `C:\Projects\caura-ai\yanki-AMB-benchmark\outputs\personamem\caura-bulk-2k-top50\rag\32k.json`;
- 503/589, 85.4%;
- raw bulk ingest;
- 2k-character chunks;
- `top_k=50`;
- 16.3K average answer-context tokens;
- Gemini 3.8 Flash answerer;
- exact MCQ-letter scoring; Gemini 2.5 Flash-Lite is recorded as `judge_llm` but is not invoked for PersonaMem.

What worked:

- raising the server's retrieval ceiling to allow a real 2k-chunk candidate set;
- balancing selectivity and coverage at `top_k=50`;
- using lexical signal rather than pure vector only;
- retaining raw detail for fact and preference-history questions.

What did not reliably work:

- assuming more context is always better;
- assuming a "Pro" reader is better than Flash;
- relying on extracted facts alone for verbatim and longitudinal details;
- treating the old dedup-loss diagnosis as current—the major August loss was an idempotency bug and is fixed.

The remaining weak category is `suggest_new_ideas`, which is strongly answer-model and prompt bound. Server retrieval work should not be credited with solving it.

### LongMemEval

Meaningful local sequence:

- generic pack-v5 run: 41/54, 75.9%;
- non-adaptive `top_k=50`, `fts_weight=0.6`, two-query run: 44/54, 81.5%;
- category-adaptive v4: 47/54, 87.0%;
- corrected adaptive 200-question run: 171/200, 85.5%;
- official-mix projection from the 200-question per-type rates: about 82.4%.

The 200-question category results were:

- single-session user: 32/34, 94.1%;
- single-session assistant: 33/34, 97.1%;
- knowledge update: 31/34, 91.2%;
- preference: 25/30, 83.3%;
- temporal: 26/34, 76.5%;
- multi-session: 24/34, 70.6%.

Median reader context was roughly 128.8K characters. Retrieval p50/p95 was 3.93/10.03 seconds; generation p50/p95 was 2.54/9.40 seconds.

Main lesson: the high-scoring categories do not need more retrieval. Multi-session and temporal need better evidence organization and reasoning. All six analyzed v4 failures already contained every annotated evidence session.

Protocol caveat: category-adaptive `top_k=15–60` and query count are chosen from the gold `question_type`. Keep 85.5% as an optimization result. Build a query-only intent classifier before claiming the adaptive policy as production behavior.

The attempted full run is not a result: `caura-full-eval-v1` contains 468 hypotheses but no `eval_results.json`. Do not extrapolate or headline it.

### BEAM

Current 100K development artifact:

- output: `C:\Projects\caura-ai\yanki-beam\outputs\caura-100k-harness_72ba6b53\beam_results_20260909_162921.json`;
- 200 questions, 20 per ability;
- Gemini 3.8 Flash answerer;
- Gemini 3.5 Flash-Lite judge;
- one Caura query;
- top-10/20/50/200 cutoffs.

At `top_200`:

- 94/200 passed, 47.0%;
- average nugget score 0.450;
- about 120.9K answer-prompt tokens;
- summarization 0/20;
- event ordering 1/20;
- multi-session reasoning 4/20.

At `top_20`:

- 98/200 passed, 49.0%;
- average nugget score 0.492;
- about 23.4K answer-prompt tokens.

This result is labeled `protocol=comparable`, but its metadata uses Gemini rather than the GPT-4o answerer and judge required by the repository's comparable protocol definition. It is therefore a development run, not a Mem0-comparable run.

The previous 200-question Caura run scored 41.0% at `top_200`; the newer harness scored 47.0%. Because harness and context handling changed, this is evidence of protocol sensitivity, not a measured six-point Caura server gain.

The highest-value BEAM work is not a 1M run yet. First fix protocol metadata, date semantics, session coverage, summarization memory, and event ordering. Then run the first publishable paired Caura/Mem0 1M evaluation with GPT-4o, `top_200`, the same 100 conversations, and both pass rate and average score.

### LoCoMo

Newest held-out artifact:

- output: `C:\Projects\caura-ai\yanki-locomo\outputs\caura-agentic-v1-holdout50-seed23-20260910\results.json`;
- pinned LoCoMo dataset SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`;
- 50 questions selected with seed 23, balanced at 10 per category and excluding all 50 questions from the earlier `caura-flash38-50-k30-20260910` development result;
- production bulk-strong ingestion, 4k-character chunks, one isolated agent/fleet per conversation;
- memories reused from `caura-balanced-50-v3-20260910`;
- `top_k=30`, one literal query, no multiquery;
- Gemini 3.8 Flash reader and judge, temperature zero;
- agentic stages: evidence extraction, answer, conditional inference, and context-only verification.

Results:

- answerable semantic accuracy: **33/40, 82.5%**; Wilson 95% CI **68.0–91.3%**;
- multi-hop: 8/10; temporal: 7/10; open-domain: 9/10; single-hop: 9/10;
- adversarial abstention: **5/10, 50%**, reported separately;
- lexical F1: 0.538;
- annotated-evidence hit rate: **100%**;
- annotated-evidence recall: **99.2%**;
- retrieval p50/p95: 1.44/3.58 seconds;
- generation p50/p95: 5.37/10.12 seconds.

The paired direct run used the same held-out questions, memories, retrieval settings, reader, and judge. It scored:

- answerable semantic accuracy: **31/40, 77.5%**;
- adversarial abstention: **10/10, 100%**;
- lexical F1: 0.620;
- generation p50/p95: 1.69/4.31 seconds.

The comparison is unusually informative. Agentic orchestration gained five observed semantic-accuracy points without changing stored memory, but it was roughly three times slower at p50, lowered lexical F1, and lost five abstention decisions. The wide, overlapping confidence intervals also mean 40 answerable questions are insufficient to establish a stable accuracy gain. The next run should freeze both pipelines and evaluate the full 1,540 category 1–4 questions, with all 446 adversarial questions retained as a separate abstention panel.

Retrieval is no longer the dominant issue on this sample: nearly every annotated evidence turn reached the reader. The remaining errors primarily test reasoning, answer control, and abstention behavior. This supports the broader conclusion from LongMemEval that once evidence recall is near saturation, adding more `top_k` is less valuable than better evidence use.

The previous 77.6% April 2026 claim remains a legacy directional result because its end-to-end artifact is absent. It should no longer be presented as Caura's current LoCoMo baseline.

## 6. Recommended Caura engineering roadmap

### P0: protocol and temporal correctness

#### Freeze two evaluation tracks

Create:

- **Comparable track:** no gold labels, official split, fixed answerer/judge, one declared retrieval mode, failures counted wrong.
- **Research track:** adaptive retrieval, alternate readers/judges, query-specific policies, and architecture ablations.

Every result should include a machine-readable manifest with dataset hash, question IDs, harness commit, server release, tenant settings, ingestion mode, all retrieval parameters, model versions, prompts, token counts, failure counts, and latency percentiles.

#### Forward and honor `valid_at`

In the LongMemEval adapter, forward `question_date` as `/search.valid_at`.

In first-stage scoring:

- compute age relative to `valid_at` when supplied;
- use `ts_valid_start` as the primary event anchor;
- use `created_at` only when event time is unavailable;
- apply temporal windows and valid-end currency relative to the same reference.

Add regression tests with old events ingested today and a historical query date.

#### Harden run isolation

- include protocol and run hash in every agent/fleet ID;
- preflight expected memory counts before `--skip-ingest`;
- refuse resume when the manifest differs;
- record all failed ingests and searches;
- prevent duplicate hypotheses by question ID;
- purge and verify scope before a fresh run.

#### Stop sending ignored write hints

The adapters send `?mode=strong` to bulk write, while the bulk API's meaningful control is item-level `write_mode`. Set the supported field explicitly or remove the query parameter. The server should reject unknown behavior-changing parameters rather than silently accepting them.

### P1: memory architecture

#### Add coverage-aware retrieval intents

Detect query intents such as:

- count or total;
- summarize all;
- order events;
- compare old and new;
- current preference;
- historical preference;
- abstention.

For count, summary, ordering, and multi-session queries, diversify by source/session/date before reranking. Similarity-only top-k tends to return several near-duplicate chunks from one session.

The classifier must use query text, not benchmark labels. Log its inferred intent for analysis.

#### Maintain multiple granularities

Use four complementary representations:

1. raw source chunks for verbatim evidence;
2. atomic facts for precise retrieval;
3. session summaries for coverage;
4. longitudinal state or entity summaries for evolution.

Caura already has atomic-fact fan-out wired into enrichment. The next step is to measure when it runs on bulk benchmark writes, preserve source/session links on child facts, and retrieve across layers without flooding the reader with duplicates.

#### Build preference and constraint state

Represent:

- positive preferences;
- dislikes and negative constraints;
- current value;
- prior values;
- reason for change;
- valid time;
- confidence and provenance.

Retrieve the current state plus the relevant change chain. This directly targets PersonaMem preference evolution, LongMemEval preference and knowledge update, and BEAM preference/instruction/contradiction categories.

#### Add hierarchical summaries

BEAM summarization cannot be solved by returning isolated high-similarity chunks. Create bounded session summaries and rolling longitudinal summaries with provenance links. Retrieval should return summaries for broad questions and raw evidence for verification.

#### Make ordering first-class

Preserve stable turn and session sequence IDs. Dates alone are insufficient when events share a date or have relative references. Event-ordering responses should be generated from an explicitly ordered evidence set, not inferred from relevance rank.

### P2: ranking and diagnostics

#### Run controlled server A/B tests

The AMB sweep justifies full paired tests of:

- `fts_weight=0.6`;
- unified `score_formula=1`;
- larger candidate pools;
- second-stage reranking;
- server-side query rewriting;
- source/session diversity.

Change one variable at a time, reuse fixed hypotheses only for judge tests, and report confidence intervals or repeated-run variance.

#### Improve diagnostics

Return or log:

- server timing by stage;
- inferred retrieval intent;
- `valid_at` and effective temporal anchor;
- candidate counts before and after each filter;
- dense, lexical, temporal, entity, graph, status, and reranker score components;
- number of distinct sources/sessions in the result;
- unknown or empty agent-scope warnings.

#### Evaluate native `/recall` separately

Caura's `/recall` already sorts by valid time, adds the current date when `valid_at` is provided, and performs grounded step-by-step synthesis. Test it in AMB native-answer or agentic mode, not as a silent replacement inside the shared RAG track.

## 7. Experimental plan

### Phase 1: one week

- freeze manifests for all four local harnesses;
- remove benchmark-label access from comparable LongMemEval;
- forward `valid_at`;
- fix BEAM protocol enforcement so metadata cannot say comparable with the wrong models;
- add run-scope preflight and resume guards;
- rejudge a fixed sample with the canonical judge plus a second judge and human audit.

### Phase 2: two weeks

- implement temporal-reference scoring tests and fix;
- add query-only retrieval intent classification;
- add session/source diversity;
- run paired LongMemEval 500 and AMB 589 A/Bs;
- report evidence-session recall beside answer accuracy.

### Phase 3: two to four weeks

- add session and longitudinal summaries;
- add explicit preference/constraint state;
- add ordered event evidence;
- run BEAM 100K architecture ablations;
- only then run paired BEAM 1M comparable evaluations.

## 8. Required reporting checklist

A publishable result must state:

- benchmark version, split, scale, and question count;
- category exclusions and weighting;
- memory-system and harness commits;
- server release and tenant feature flags;
- ingestion granularity and extraction model;
- embedding, reranking, graph, and lexical settings;
- `top_k`, candidate pool, query count, merge method, and context tokens;
- sort and packing policy;
- date fields and query reference behavior;
- answer model and full prompt version;
- judge or deterministic scorer, full prompt/code version, and parse-failure behavior;
- canonical metric, including Kendall tau-b versus nugget scoring for BEAM event ordering;
- retries, failures, resume, and cache policy;
- accuracy by category;
- retrieval evidence recall where annotations exist;
- token use, ingest cost/time, and retrieval p50/p95;
- whether the run used benchmark labels, oracle evidence, agentic search, or native answer;
- confirmation that answers, evidence IDs, `has_answer`, and rubrics were excluded from ingestion;
- whether competitor numbers were locally rerun or copied from a publication.

## Conclusion

Caura does not need to chase one magic `top_k`. The trial record shows that scores improve when storage granularity, retrieval coverage, context budget, reader behavior, and judge protocol are aligned.

The clearest product gaps are temporal reference semantics, cross-session coverage, longitudinal preference state, summarization, and event ordering. The clearest evaluation gap is protocol discipline.

If engineering fixes those two sets together, Caura can improve real memory behavior and produce benchmark numbers that remain credible when another team reruns them.

## Internal evidence

- AMB summary: `C:\Projects\caura-ai\yanki-AMB-benchmark\caura-RESULTS.md`
- AMB engineering audit: `C:\Projects\caura-ai\yanki-AMB-benchmark\CAURA-ENGINEERING-NOTE-2026-09-06.md`
- AMB result artifacts: `C:\Projects\caura-ai\yanki-AMB-benchmark\outputs\personamem\`
- LongMemEval corrected result: `C:\Projects\caura-ai\yanki-longmemeval\outputs\caura-200-adaptive-v1\eval_results.json`
- LongMemEval repair record: `C:\Projects\caura-ai\yanki-longmemeval\outputs\caura-200-adaptive-v1\baseline_repair.json`
- LongMemEval judge variants: `C:\Projects\caura-ai\yanki-longmemeval\outputs\caura-50-adaptive-v4\`
- BEAM result: `C:\Projects\caura-ai\yanki-beam\outputs\caura-100k-harness_72ba6b53\beam_results_20260909_162921.json`
- LoCoMo held-out agentic result: `C:\Projects\caura-ai\yanki-locomo\outputs\caura-agentic-v1-holdout50-seed23-20260910\results.json`
- LoCoMo paired direct result: `C:\Projects\caura-ai\yanki-locomo\outputs\caura-direct-holdout50-seed23-20260910\results.json`
- Caura benchmark notes: `C:\Projects\caura-ai\caura\BENCHMARKS.md`
- Caura temporal scoring: `C:\Projects\caura-ai\caura\core-storage-api\src\core_storage_api\services\postgres_service.py`
- Caura grounded recall: `C:\Projects\caura-ai\caura\core-api\src\core_api\services\recall_service.py`
