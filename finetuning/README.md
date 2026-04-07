# Finetuning — Open-Source Post-Training and Adapter Tuning

**A notebook-based course on modern open-source finetuning, alignment, and post-training for 2026.**

This folder is scaffolded as a 22-notebook curriculum focused on the practical path most teams can actually run: adapter-based supervised finetuning first, preference alignment second, and reinforcement learning only after strong data and evaluation foundations are in place. The emphasis is on transparent, reproducible workflows using open models, Google Colab Pro, and the Unsloth-centered training stack.

| Component | Implementation |
|---|---|
| **Training Stack** | Unsloth + TRL + PEFT |
| **Default Recipe** | 4-bit QLoRA adapters |
| **Model Class** | Open instruct models in the 3B–8B range |
| **Runtime** | Google Colab Pro (T4/L4 first, A100 optional when available) |
| **Artifacts** | Adapters, merged checkpoints, eval reports, export-ready weights |
| **Framework** | Notebook-first, open-source only, minimal abstractions |

> **Prerequisites:** Complete [prompt-engineering/](../prompt-engineering/), [rag/](../rag/), [agents/](../agents/), and [evals/](../evals/) first. This track assumes you already know how to control models, ground them, orchestrate them, and measure them.
> **Runtime:** Google Colab Pro is the default target. Most notebooks are designed around QLoRA on Colab-class GPUs, with heavier alignment and RL notebooks framed conservatively for T4/L4 runtimes.

---

## Why This Track Exists

The first four Castalia tracks teach how to:
- control models with prompts
- ground them with retrieval
- orchestrate them with tools and agent loops
- evaluate whether changes actually improve quality

This track teaches what comes next: how to change the model itself.

In other words:
- `prompt-engineering/` teaches **model control**
- `rag/` teaches **knowledge grounding**
- `agents/` teaches **action and orchestration**
- `evals/` teaches **measurement and reliability**
- `finetuning/` teaches **post-training and behavior change**

That ordering matters. Finetuning without evals becomes guesswork. Finetuning after evals becomes disciplined model improvement: build a baseline, collect data, run adapter training, measure gains, inspect regressions, and decide whether more training is justified.

---

## Course Structure

### Module 1: Foundations and Environment (Notebooks 01–04)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_intro_to_finetuning_2026.ipynb` | Finetuning vs prompting vs RAG, where post-training fits in the 2026 stack, when not to train |
| 02 | `02_colab_pro_and_unsloth_setup.ipynb` | Colab Pro setup, Unsloth installation, runtime verification, common environment failures |
| 03 | `03_model_selection_and_vram_budgeting.ipynb` | Choosing 3B/4B/7B/8B models, VRAM budgeting, QLoRA vs LoRA vs dense tuning |
| 04 | `04_datasets_chat_templates_and_loss_masking.ipynb` | Chat formatting, assistant-only loss, completion masking, train/validation split discipline |

### Module 2: Supervised Fine-Tuning with Adapters (Notebooks 05–09)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 05 | `05_first_qlora_sft_run.ipynb` | End-to-end first QLoRA run with Unsloth, TRL SFT, and notebook-native evaluation |
| 06 | `06_data_curation_cleaning_and_splitting.ipynb` | Data cleaning, deduplication, leakage checks, curriculum shaping, realistic validation sets |
| 07 | `07_lora_hyperparameters_and_target_modules.ipynb` | Rank, alpha, dropout, target-module choices, update scope, trade-offs in adapter capacity |
| 08 | `08_packing_long_context_and_throughput.ipynb` | Sequence packing, context length trade-offs, throughput tuning, Colab-friendly batching |
| 09 | `09_adapter_inspection_merging_and_export.ipynb` | Inspecting adapters, merge vs keep-separate decisions, export and inference handoff |

### Module 3: Preference Alignment (Notebooks 10–13)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 10 | `10_preference_data_construction.ipynb` | Building chosen/rejected datasets, rubric-derived preferences, quality control for alignment data |
| 11 | `11_dpo_alignment.ipynb` | Direct Preference Optimization as the default second stage after SFT |
| 12 | `12_orpo_and_kto_alignment.ipynb` | ORPO and KTO, when reference-free or binary-preference formulations are a better fit |
| 13 | `13_finetuning_evaluation_and_regressions.ipynb` | Compare base vs tuned models, regression analysis, task-level gains, failure bucket reviews |

### Module 4: Domain Adaptation and Data Scaling (Notebooks 14–16)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 14 | `14_continued_pretraining.ipynb` | When domain-adaptive continued pretraining is justified, and how to scope it safely |
| 15 | `15_synthetic_data_and_distillation.ipynb` | Synthetic instruction generation, filtering, teacher-student distillation, data amplification |
| 16 | `16_forgetting_mixture_design_and_safety.ipynb` | Mixture design, catastrophic forgetting, retention checks, safety-aware post-training |

### Module 5: RL and Reasoning Post-Training (Notebooks 17–19)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 17 | `17_grpo_foundations_and_reward_design.ipynb` | GRPO fundamentals, verifier-driven rewards, why RL is powerful but not the default |
| 18 | `18_reasoning_finetuning_with_grpo.ipynb` | Small-scale reasoning and verifiable-task RL experiments on Colab-class hardware |
| 19 | `19_rl_for_tool_use_and_structured_outputs.ipynb` | Rewarding tool correctness, schema adherence, and constrained-output behavior |

### Module 6: Capstone — Castalia Mentor (Notebooks 20–22)

| # | Notebook | What You'll Build |
|---|----------|-------------------|
| 20 | `20_project_dataset_pipeline.ipynb` | A Castalia tutoring dataset assembled from prior course concepts and evaluation rubrics |
| 21 | `21_project_training_pipeline.ipynb` | The full post-training pipeline: SFT first, preference alignment second, optional RL extension |
| 22 | `22_project_benchmark_and_export.ipynb` | Final benchmark, regression review, adapter export, and deployment-ready handoff |

---

## Design Philosophy

This course follows four core principles:

1. **Adapter-First**: The default move is not dense finetuning. It is adapter tuning that is cheaper, faster to iterate, easier to evaluate, and easier to roll back.

2. **QLoRA First**: On Colab Pro, 4-bit loading plus LoRA adapters is the practical default. Students should learn the workflow that fits real hardware constraints before touching expensive alternatives.

3. **Preference Alignment Second**: SFT establishes the initial behavior. DPO, ORPO, and KTO refine quality, style, and preference structure after that baseline exists.

4. **RL Later**: RL belongs after good data pipelines, good evaluators, and tasks with useful reward signals. It is an advanced tool, not the first lever to pull.

The underlying research lesson is simple: data quality and evaluation discipline usually matter more than clever training tricks. This track therefore treats curation, formatting, validation, and regression analysis as first-class training topics.

---

## Dependency Philosophy

- **Unsloth at the center**: fast Colab-friendly training for LoRA, QLoRA, continued pretraining, preference optimization, and RL-style post-training
- **TRL for training algorithms**: SFT, DPO, ORPO, KTO, and GRPO-style workflows live here when the curriculum moves beyond the first adapter run
- **PEFT for adapter lifecycle**: configuration, saving, loading, merging, and export of parameter-efficient adapters
- **Reuse the Hugging Face core stack**: `transformers`, `datasets`, `accelerate`, `bitsandbytes`, and `torch`
- **Minimal supporting packages**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- **Standard library first**: `json`, `csv`, `math`, `random`, `statistics`, `pathlib`, `dataclasses`, `typing`, `collections`
- **Explicitly avoid opaque tuning platforms**: no managed SaaS finetuning products, no hidden trainer layers, no vendor-specific post-training dashboards

The goal is to keep every notebook understandable from end to end. Students should know what is happening in the dataset, trainer, adapter, and evaluation loop instead of outsourcing understanding to a platform.

---

## Recommended Models and Hardware

The hardware target is **Google Colab Pro**, so the model strategy is intentionally conservative:

- **Default models:** 3B–8B instruct checkpoints from open families such as Qwen, Gemma, and Llama-style releases
- **Best default path:** QLoRA adapters on 4-bit weights
- **T4-friendly work:** 3B/4B models, short-to-moderate context windows, SFT and smaller preference datasets
- **L4-friendly work:** 7B/8B QLoRA, longer contexts, smoother preference optimization, more comfortable experiment iteration
- **A100 when available:** optional for heavier continued pretraining experiments, larger ablations, or more ambitious RL notebooks
- **Dense finetuning framing:** taught conceptually and experimentally, but not positioned as the default student workflow

This keeps the course honest: most learners do not have multi-GPU clusters, so the default recipes should fit the machines they actually have access to.

---

## Castalia Mentor Capstone

The overarching project for this track is **Castalia Mentor**: a specialized open-source tutor model for the Castalia curriculum.

Across the capstone sequence, students will:
- assemble a tutoring dataset grounded in concepts from the earlier folders
- train an initial supervised adapter using Unsloth
- improve it with preference alignment
- optionally test a small RL stage for verifier-friendly tasks
- compare the tuned model against the base model using the evaluation mindset established in `evals/`
- export the resulting adapter or merged artifact for later use

The point of the capstone is not merely to train a model. It is to build a training pipeline that is measurable, inspectable, and connected to the rest of the repository.

---

## How to Use This Course

### Recommended Path After the Existing Tracks
1. Finish [prompt-engineering/](../prompt-engineering/) to learn control, decomposition, and prompting strategy
2. Finish [rag/](../rag/) to learn grounding, retrieval, and evidence-aware generation
3. Finish [agents/](../agents/) to learn tools, orchestration, memory, and system design
4. Finish [evals/](../evals/) to learn measurement, benchmarking, and regression discipline
5. Work through `finetuning/` in order: Module 1 → Module 2 → Module 3 → Module 4 → Module 5 → Module 6

### What This Track Adds
- It turns measured system behavior into trainable model behavior
- It explains when finetuning is better than prompting or RAG, and when it is not
- It teaches a realistic 2026 open-source stack rather than idealized cluster-scale training
- It connects post-training decisions back to benchmark evidence instead of anecdotal wins

### What Comes After `finetuning/`
After this track, learners should be prepared for the next layer of the stack:
- inference optimization and systems engineering
- the dedicated [multimodal/](../multimodal/) track
- multimodal post-training
- larger-scale distributed training
- deployment, monitoring, and continual improvement loops

---

## Course Status

The `finetuning/` course is now fully created as a 22-notebook track. The folder contains the complete notebook sequence, from finetuning foundations through supervised finetuning, preference alignment, continued pretraining, RL-style post-training, and the three-part Castalia Mentor capstone. This README now documents the completed structure and design philosophy of the finished course.
