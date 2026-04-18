# 📚 EduTutor Learning Path — Castalia Prerequisites

Before each EduTutor notebook, work through the listed Castalia notebooks to build the skills you'll need. Each section explains *why* that notebook matters for the hackathon work.

---

## Before Notebook 1: Dataset Generation

> **Goal:** Understand how to craft high-quality training data for LLMs, including persona prompting, chat templates, and synthetic data generation.

### Required Reading (in order)

| # | Castalia Notebook | Why You Need It |
|---|---|---|
| 1 | [prompt-engineering/intro-prompt-engineering-lesson.ipynb](../../../prompt-engineering/intro-prompt-engineering-lesson.ipynb) | Foundation — understand how prompts shape LLM behavior |
| 2 | [prompt-engineering/role-prompting.ipynb](../../../prompt-engineering/role-prompting.ipynb) | EduTutor's persona is built via role prompting — learn how to define consistent character behavior |
| 3 | [prompt-engineering/few-shot-learning.ipynb](../../../prompt-engineering/few-shot-learning.ipynb) | Our data generation prompts use few-shot examples to demonstrate "good" vs "bad" tutor behavior |
| 4 | [prompt-engineering/task-decomposition-prompts.ipynb](../../../prompt-engineering/task-decomposition-prompts.ipynb) | Breaking complex generation tasks into manageable pieces — critical for generating multi-turn conversations |
| 5 | [foundations/00_how_llms_work.ipynb](../../../foundations/00_how_llms_work.ipynb) | Understand tokenization, sampling, and generation so you know what's happening when the model produces training data |
| 6 | [foundations/03_sampling_and_decoding.ipynb](../../../foundations/03_sampling_and_decoding.ipynb) | Temperature, top-p, top-k — you'll use these to control diversity in generated data |
| 7 | [finetuning/04_datasets_chat_templates_and_loss_masking.ipynb](../../../finetuning/04_datasets_chat_templates_and_loss_masking.ipynb) | **Critical** — learn how chat templates work, what loss masking is, and why you only train on assistant turns |
| 8 | [finetuning/15_synthetic_data_and_distillation.ipynb](../../../finetuning/15_synthetic_data_and_distillation.ipynb) | The core technique for NB1 — using a stronger model to generate training data for a smaller model |
| 9 | [finetuning/06_data_curation_cleaning_and_splitting.ipynb](../../../finetuning/06_data_curation_cleaning_and_splitting.ipynb) | Quality filtering, deduplication, and train/val splitting — applied to our generated dataset |

### What You'll Understand After
- How to design system prompts that create consistent personas
- How chat templates structure multi-turn conversations for training
- How to generate diverse synthetic data and filter for quality
- Why loss masking matters (we only want the model to learn tutor responses, not student messages)

---

## Before Notebook 2: Unsloth Fine-Tuning

> **Goal:** Understand the full fine-tuning pipeline — from model loading and LoRA configuration to SFT, DPO alignment, and model export.

### Required Reading (in order)

| # | Castalia Notebook | Why You Need It |
|---|---|---|
| 1 | [finetuning/01_intro_to_finetuning_2026.ipynb](../../../finetuning/01_intro_to_finetuning_2026.ipynb) | Understand when to fine-tune vs prompt vs RAG — and why we chose fine-tuning for EduTutor |
| 2 | [finetuning/02_colab_pro_and_unsloth_setup.ipynb](../../../finetuning/02_colab_pro_and_unsloth_setup.ipynb) | **Critical** — set up Unsloth, verify GPU, understand the runtime environment |
| 3 | [finetuning/03_model_selection_and_vram_budgeting.ipynb](../../../finetuning/03_model_selection_and_vram_budgeting.ipynb) | Why we chose E4B, how QLoRA saves VRAM, the trade-offs between model sizes |
| 4 | [finetuning/05_first_qlora_sft_run.ipynb](../../../finetuning/05_first_qlora_sft_run.ipynb) | **Critical** — your first hands-on fine-tuning run. EduTutor NB2 extends this directly |
| 5 | [finetuning/07_lora_hyperparameters_and_target_modules.ipynb](../../../finetuning/07_lora_hyperparameters_and_target_modules.ipynb) | Understand rank, alpha, dropout, and why we target all attention + MLP modules |
| 6 | [finetuning/10_preference_data_construction.ipynb](../../../finetuning/10_preference_data_construction.ipynb) | How to build chosen/rejected pairs — directly maps to our DPO dataset |
| 7 | [finetuning/11_dpo_alignment.ipynb](../../../finetuning/11_dpo_alignment.ipynb) | **Critical** — understand DPO loss, beta tuning, and why it prevents answer-giving behavior |
| 8 | [finetuning/09_adapter_inspection_merging_and_export.ipynb](../../../finetuning/09_adapter_inspection_merging_and_export.ipynb) | How to merge LoRA adapters and export to GGUF — needed for deployment |
| 9 | [finetuning/16_forgetting_mixture_design_and_safety.ipynb](../../../finetuning/16_forgetting_mixture_design_and_safety.ipynb) | Understand catastrophic forgetting — our model must keep general knowledge while learning the tutor persona |

### Optional Deep Dives

| # | Castalia Notebook | Why |
|---|---|---|
| A | [finetuning/08_packing_long_context_and_throughput.ipynb](../../../finetuning/08_packing_long_context_and_throughput.ipynb) | Packing short conversations for faster training |
| B | [finetuning/12_orpo_and_kto_alignment.ipynb](../../../finetuning/12_orpo_and_kto_alignment.ipynb) | Alternative alignment methods if DPO doesn't work well |
| C | [finetuning/17_grpo_foundations_and_reward_design.ipynb](../../../finetuning/17_grpo_foundations_and_reward_design.ipynb) | Advanced — RL-based training if you want the tutor to learn from student outcomes |

### What You'll Understand After
- The full SFT → DPO two-stage pipeline and why each stage matters
- How LoRA adapters work and how to configure them
- How to prevent the model from forgetting general knowledge
- How to export a fine-tuned model for local deployment

---

## Before Notebook 3: Evaluation Harness

> **Goal:** Understand how to rigorously measure whether your fine-tuned model is actually better, using LLM-as-Judge, rubrics, and A/B testing.

### Required Reading (in order)

| # | Castalia Notebook | Why You Need It |
|---|---|---|
| 1 | [evals/01_evals_why_measurement_matters.ipynb](../../../evals/01_evals_why_measurement_matters.ipynb) | Foundation — why "it seems better" is not enough, why we need systematic evaluation |
| 2 | [evals/02_building_eval_datasets.ipynb](../../../evals/02_building_eval_datasets.ipynb) | How to build held-out test sets that are truly independent from training data |
| 3 | [evals/03_metrics_from_scratch.ipynb](../../../evals/03_metrics_from_scratch.ipynb) | Understand precision, recall, F1 — the vocabulary of measurement |
| 4 | [evals/04_error_analysis_and_failure_buckets.ipynb](../../../evals/04_error_analysis_and_failure_buckets.ipynb) | Categorizing failures — we need to know WHERE the model fails (Red zone? Misconceptions?) |
| 5 | [evals/06_rubrics_and_llm_as_judge.ipynb](../../../evals/06_rubrics_and_llm_as_judge.ipynb) | **Critical** — the core technique for NB3. How to design rubrics and use an LLM to score outputs |
| 6 | [evals/07_pairwise_and_preference_eval.ipynb](../../../evals/07_pairwise_and_preference_eval.ipynb) | A/B comparison methodology — how we compare EduTutor vs base Gemma 4 |
| 7 | [evals/19_safety_and_policy_evals.ipynb](../../../evals/19_safety_and_policy_evals.ipynb) | Child safety checks — ensuring the model never produces harmful content |
| 8 | [finetuning/13_finetuning_evaluation_and_regressions.ipynb](../../../finetuning/13_finetuning_evaluation_and_regressions.ipynb) | Regression testing — making sure fine-tuning didn't break general capabilities |

### What You'll Understand After
- How to design custom rubrics for non-standard evaluation dimensions (like emotional co-regulation)
- How LLM-as-Judge works, its biases, and how to mitigate them
- How to run A/B comparisons and compute statistically meaningful win rates
- How to identify failure modes and iterate on training data

---

## Before Notebook 4: Agentic Tutor UI

> **Goal:** Understand how to build agentic systems with memory, tool use, state management, and human-in-the-loop interaction.

### Required Reading (in order)

| # | Castalia Notebook | Why You Need It |
|---|---|---|
| 1 | [agents/01_intro_to_agentic_ai.ipynb](../../../agents/01_intro_to_agentic_ai.ipynb) | Foundation — what agents are, the perception-action loop, agentic levels L0–L4 |
| 2 | [agents/02_the_agent_loop.ipynb](../../../agents/02_the_agent_loop.ipynb) | **Critical** — the universal agent loop that EduTutor's architecture is built on |
| 3 | [agents/03_tool_use_and_function_calling.ipynb](../../../agents/03_tool_use_and_function_calling.ipynb) | How tools work — EduTutor uses flashcards, scaffolding hints, brain breaks, and difficulty adjustment tools |
| 4 | [agents/04_structured_output_parsing.ipynb](../../../agents/04_structured_output_parsing.ipynb) | Parsing JSON from LLM outputs — needed for the zone classifier and state updates |
| 5 | [agents/05_building_a_react_agent.ipynb](../../../agents/05_building_a_react_agent.ipynb) | **Critical** — EduTutor uses a ReAct (Thought → Action → Observation) loop |
| 6 | [agents/10_agent_memory_short_term.ipynb](../../../agents/10_agent_memory_short_term.ipynb) | Conversation memory management — EduTutor tracks the last N turns for context |
| 7 | [agents/07_reflection_and_self_critique.ipynb](../../../agents/07_reflection_and_self_critique.ipynb) | The tutor reflects on whether the student understood — this technique could enhance responses |
| 8 | [agents/24_agent_safety_and_guardrails.ipynb](../../../agents/24_agent_safety_and_guardrails.ipynb) | **Critical** — child safety guardrails, PII prevention, content filtering |
| 9 | [agents/25_human_in_the_loop.ipynb](../../../agents/25_human_in_the_loop.ipynb) | Teacher/parent oversight integration — escalation when the agent detects a crisis |

### Optional Deep Dives

| # | Castalia Notebook | Why |
|---|---|---|
| A | [agents/22_shared_state_and_blackboard.ipynb](../../../agents/22_shared_state_and_blackboard.ipynb) | Advanced state management patterns for the student state machine |
| B | [agents/27_cost_and_latency_optimization.ipynb](../../../agents/27_cost_and_latency_optimization.ipynb) | Optimize inference speed for real-time chat |
| C | [agents/28_error_handling_and_resilience.ipynb](../../../agents/28_error_handling_and_resilience.ipynb) | Graceful failure handling when the model produces bad output |

### What You'll Understand After
- How to build agents that maintain state across turns
- How to integrate tools into an LLM workflow
- How to build safe, guardrailed systems for vulnerable users (children)
- How to create interactive UIs that expose agent internals

---

## Suggested Study Schedule

| Day | Study (Castalia) | Build (EduTutor) |
|---|---|---|
| **Day 1–2** | Prompt engineering prereqs (6 notebooks) | — |
| **Day 3–4** | Finetuning prereqs: NB 01–06 | — |
| **Day 5** | Finetuning prereqs: NB 15, 04 | Run **NB1**: Generate dataset |
| **Day 6–7** | Finetuning prereqs: NB 05, 07, 10, 11, 09, 16 | — |
| **Day 8** | — | Run **NB2**: Fine-tune model |
| **Day 9–10** | Evals prereqs (8 notebooks) | — |
| **Day 11** | — | Run **NB3**: Evaluate model |
| **Day 12–13** | Agents prereqs (9 notebooks) | — |
| **Day 14** | — | Run **NB4**: Build demo |
| **Day 15** | — | Record video, submit |

> **Total: ~37 Castalia notebooks** spread across 2 weeks, with EduTutor builds interleaved. This is aggressive but doable — each Castalia notebook takes 30-60 minutes.
