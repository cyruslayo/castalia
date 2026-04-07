# Evals — Evaluation, Benchmarking, and Reliability Engineering

**A notebook-based course on measuring, comparing, debugging, and hardening LLM systems with open-source tools.**

This folder will contain 22 Jupyter notebooks organized into 5 modules, progressing from evaluation fundamentals to a full benchmark harness for prompt systems, RAG pipelines, and agents. The goal is not just to build systems, but to prove that they work, understand where they fail, and improve them with disciplined measurement.

| Component | Implementation |
|---|---|
| **Eval Harness** | Pure Python + notebook utilities |
| **Judge Models** | Open-source instruct models from Hugging Face |
| **Retrieval Benchmarks** | sentence-transformers + FAISS |
| **Artifacts** | JSON/CSV outputs, reproducible local runs |
| **Framework** | Minimal dependencies, no vendor eval platforms |

> **Prerequisites:** Complete [prompt-engineering/](../prompt-engineering/), [rag/](../rag/), and [agents/](../agents/) first. This track sits after the trilogy and turns those building skills into measurement and reliability skills.
> **Runtime:** Google Colab friendly. Most notebooks are lightweight; judge-model and larger experiment notebooks benefit from a free Colab T4 GPU.

---

## Why This Track Exists

The first three Castalia folders teach how to:
- control models with prompts
- ground them with retrieval
- orchestrate them with tools and multi-agent patterns

This track teaches what comes next: how to evaluate whether those systems are actually getting better.

Without evals, every prompt tweak, RAG change, or agent redesign is mostly anecdotal. With evals, you can run ablations, measure regressions, inspect failures, compare variants, and make reliability a first-class engineering concern.

In other words:
- `prompt-engineering/` teaches **model interaction**
- `rag/` teaches **knowledge grounding**
- `agents/` teaches **action and orchestration**
- `evals/` teaches **measurement, benchmarking, and iteration discipline**

---

## Course Structure

### Module 1: Evaluation Foundations (Notebooks 01–04)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_evals_why_measurement_matters.ipynb` | Offline vs online evals, benchmark design, why anecdotal testing is misleading |
| 02 | `02_building_eval_datasets.ipynb` | Golden sets, realistic task sampling, label formats, splits, leakage concerns |
| 03 | `03_metrics_from_scratch.ipynb` | Exact match, overlap metrics, precision/recall/F1, ranking metrics, metric limits |
| 04 | `04_error_analysis_and_failure_buckets.ipynb` | Failure taxonomies, severity scoring, recurring error patterns, turning failures into roadmap decisions |

### Module 2: Prompt Evaluation (Notebooks 05–08)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 05 | `05_rule_based_prompt_eval.ipynb` | Deterministic scorers, schema checks, regex grading, cheap regression tests |
| 06 | `06_rubrics_and_llm_as_judge.ipynb` | Rubric design, judge prompts, calibration, bias, judge-model limitations |
| 07 | `07_pairwise_and_preference_eval.ipynb` | A/B comparisons, pairwise ranking, win rates, uncertainty-aware comparison |
| 08 | `08_prompt_experiment_design.ipynb` | Prompt ablations, temperature controls, sample sizing, avoiding misleading wins |

### Module 3: RAG Evaluation (Notebooks 09–12)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 09 | `09_retrieval_metrics.ipynb` | recall@k, precision@k, MRR, nDCG intuition, relevance judgments |
| 10 | `10_faithfulness_and_groundedness.ipynb` | Support checking, quote alignment, hallucination detection, unsupported-claim scoring |
| 11 | `11_citation_and_evidence_coverage.ipynb` | Citation completeness, evidence coverage, missing-source detection, claim-to-source mapping |
| 12 | `12_rag_ablation_lab.ipynb` | Chunking, embeddings, reranking, context budgeting, end-to-end benchmark tables |

### Module 4: Agent Evaluation (Notebooks 13–17)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 13 | `13_tool_use_and_task_success_eval.ipynb` | Tool selection accuracy, argument correctness, task completion metrics |
| 14 | `14_trajectory_grading.ipynb` | Step-level inspection, trace audits, termination quality, loop and stall detection |
| 15 | `15_multi_agent_system_eval.ipynb` | Handoff quality, delegation quality, coordination failures, shared-state correctness |
| 16 | `16_cost_latency_reliability_eval.ipynb` | Quality vs latency, quality vs cost, timeout analysis, retry behavior |
| 17 | `17_agent_robustness_and_adversarial_eval.ipynb` | Perturbation tests, prompt injection suites, malformed tool outputs, environment stress tests |

### Module 5: Operations, Safety, and Regression (Notebooks 18–22)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 18 | `18_regression_testing_for_llm_systems.ipynb` | Benchmark snapshots, pass/fail thresholds, regression gates |
| 19 | `19_safety_and_policy_evals.ipynb` | Policy rubrics, harmful output checks, tool misuse suites, safety stress tests |
| 20 | `20_human_eval_workflows.ipynb` | Reviewer guidelines, adjudication, disagreement resolution, hybrid human/automated evaluation |
| 21 | `21_experiment_tracking_and_reporting.ipynb` | Run metadata, leaderboard tables, reproducibility logs, experiment comparisons |
| 22 | `22_castalia_bench_capstone.ipynb` | Final benchmark harness: evaluate a prompt system, a RAG system, and an agent system side by side |

---

## Design Philosophy

This course follows three core principles:

1. **Measurement Before Optimization**: We do not guess which system is better. We define tasks, datasets, scorers, and thresholds before claiming improvement.

2. **First Principles Over Platforms**: Everything is built transparently in notebooks with raw Python and simple utilities. No black-box eval dashboards, no hidden scoring logic.

3. **Reliability Over Demos**: The target is not a single impressive run. The target is a system that performs consistently across datasets, perturbations, and regression checks.

## Dependency Philosophy

- **Open-source only**: use Hugging Face models, local metrics, and notebook-native tooling
- **Reuse the Castalia stack**: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `faiss-cpu`, `numpy`
- **Standard library first**: `json`, `csv`, `math`, `statistics`, `random`, `time`, `pathlib`, `typing`, `dataclasses`, `collections`
- **Minimal extras**: only add a package when it directly improves clarity or reproducibility
- **Explicitly avoid vendor eval stacks**: no LangSmith, no Ragas, no proprietary dashboards, no closed judge APIs

The result should stay lightweight, Colab-friendly, and easy to understand from end to end.

---

## Castalia Bench Capstone

The overarching project for this track is **Castalia Bench**: a from-scratch evaluation harness that grows across the notebooks and becomes the capstone in Notebook 22.

Castalia Bench is designed to score and compare:
- prompt-only baselines
- RAG pipelines
- tool-using agents
- multi-agent workflows

Across the course, it will accumulate support for:
- dataset loading and versioned benchmark sets
- deterministic scorers and rubric-based grading
- judge-model evaluation with open-source models
- retrieval metrics and evidence-quality checks
- trajectory inspection for tool use and agent reasoning
- cost, latency, and reliability tracking
- regression suites and comparison tables

By the capstone, you should be able to benchmark one prompt system, one RAG system, and one agent system side by side, then produce a report explaining quality, failure modes, and next-step recommendations.

---

## How to Use This Course

### Recommended Path After the Trilogy
1. Finish [prompt-engineering/](../prompt-engineering/) to learn model control and decomposition
2. Finish [rag/](../rag/) to learn grounding, retrieval, and evidence-aware generation
3. Finish [agents/](../agents/) to learn tools, orchestration, memory, and multi-agent design
4. Work through `evals/` in order: Module 1 → Module 2 → Module 3 → Module 4 → Module 5

### What This Track Adds
- It converts experimentation into engineering discipline
- It gives you reusable benchmark patterns for every later system
- It creates the measurement layer needed before fine-tuning, deployment, or larger-scale production work

### After Completing `evals/`
You will be in a much stronger position to tackle the next layers of a 2026 AI systems stack:
- fine-tuning and post-training
- inference and systems engineering
- the dedicated [multimodal/](../multimodal/) track
- multimodal systems
- deeper safety, governance, and operational design

---

## Course Status

The `evals/` track is now fully populated with all 22 notebooks listed above, culminating in the Castalia Bench capstone. This folder is intended to be worked through after `prompt-engineering/`, `rag/`, and `agents/`, and it completes the measurement layer for the broader Castalia curriculum.
