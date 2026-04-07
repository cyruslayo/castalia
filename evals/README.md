# Evals — Evaluation, Benchmarking, and Reliability Engineering

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
