# Finetuning — Open-Source Post-Training and Adapter Tuning

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
