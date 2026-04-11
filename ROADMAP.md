# Castalia — Holistic Learning Roadmap

## Philosophy

Traditional linear paths teach each module in isolation: finish all of prompt engineering, then all of RAG, then agents, etc. This creates blind spots — you learn retrieval without knowing how to evaluate it, or build agents without understanding the systems that serve them.

This roadmap uses **learning orbits** — concentric spirals that weave notebooks from all 7 folders together around a shared theme. Each orbit introduces concepts at a level that reinforces and contextualizes what came before. You revisit topics (prompting, evaluation, retrieval) at increasing depth as your mental model grows.

### The 8 Folders (Domains)

| Code | Folder | Notebooks | Focus |
|------|--------|-----------|-------|
| **FN** | `foundations/` | 7 | First-principles: transformers, tokenization, embeddings, retrieval theory |
| **PE** | `prompt-engineering/` | 22 | Prompt design, templates, techniques |
| **RAG** | `rag/` | 29+ | Retrieval-augmented generation |
| **AG** | `agents/` | 37 | Agentic AI design & engineering |
| **EV** | `evals/` | 22 | Evaluation, benchmarking, reliability |
| **FT** | `finetuning/` | 22 | Post-training & adapter tuning |
| **MM** | `multimodal/` | 22 | Vision, audio, video systems |
| **SY** | `systems/` | 24 | Inference, serving, runtime engineering + practitioner guides |

**Total: ~190 notebooks across 8 domains**

> 🆕 **New in Systems**: `00_colab_pro_survival_guide` (GPU budgeting, session management, caching) and `00_debugging_playbook` (failure patterns + diagnostic toolkit across all domains). Start with these if using Colab Pro.

---

## Orbit -1 — Foundations: First Principles Before Frameworks

**Goal**: Build the conceptual bedrock that makes everything else make sense. These notebooks derive the *why* behind every tool and technique used in Orbits 0–7. Complete this orbit if you want to understand, not just use.

| # | Notebook | Domain | What You Gain |
|---|----------|--------|---------------|
| 1 | [foundations/00_how_llms_work.ipynb](foundations/00_how_llms_work.ipynb) | FN | Mental models for transformers, next-token prediction, why prompts work |
| 2 | [foundations/01_tokenization_deep_dive.ipynb](foundations/01_tokenization_deep_dive.ipynb) | FN | BPE from scratch, vocabulary effects on reasoning, token economics |
| 3 | [foundations/02_embeddings_and_vector_spaces.ipynb](foundations/02_embeddings_and_vector_spaces.ipynb) | FN | Embedding geometry, similarity metrics derived, anisotropy intuition |
| 4 | [foundations/03_sampling_and_decoding.ipynb](foundations/03_sampling_and_decoding.ipynb) | FN | Temperature, top-k/p, beam search — every knob explained from first principles |
| 5 | [foundations/04_information_retrieval_theory.ipynb](foundations/04_information_retrieval_theory.ipynb) | FN | TF-IDF, BM25, inverted indices — the retrieval fundamentals behind RAG |
| 6 | [foundations/05_attention_and_context.ipynb](foundations/05_attention_and_context.ipynb) | FN | Self-attention from scratch, KV cache, context windows, positional encodings |
| 7 | [foundations/06_scaling_laws_and_model_selection.ipynb](foundations/06_scaling_laws_and_model_selection.ipynb) | FN | Scaling laws, VRAM budgeting, quantization, model selection framework |

**Checkpoint**: You now understand *why* transformers work, *how* text becomes numbers, *what* retrieval really is, and *when* to choose which model. Every decision in the orbits ahead will make intuitive sense.

> 💡 **When to use this orbit**: Start here if you're new to AI engineering, or jump back here whenever a later notebook introduces a concept you want to understand more deeply. Each foundations notebook is cross-referenced from the orbits that depend on it.

---

## Orbit 0 — Ground Zero: What Is This Stack?

**Goal**: Build a mental map of the entire AI engineering landscape before diving deep anywhere. Understand *where* each discipline sits and *why* it exists.

| # | Notebook | Domain | What You Gain |
|---|----------|--------|---------------|
| 1 | [prompt-engineering/intro-prompt-engineering-lesson.ipynb](prompt-engineering/intro-prompt-engineering-lesson.ipynb) | PE | What prompt engineering is and why it's the foundation |
| 2 | [evals/01_evals_why_measurement_matters.ipynb](evals/01_evals_why_measurement_matters.ipynb) | EV | Why measurement must start on Day 1, not after building |
| 3 | [rag/simple_rag.ipynb](rag/simple_rag.ipynb) | RAG | Your first retrieval pipeline — see what RAG does |
| 4 | [agents/01_intro_to_agentic_ai.ipynb](agents/01_intro_to_agentic_ai.ipynb) | AG | The agentic spectrum (L0–L4), when agents add value |
| 5 | [finetuning/01_intro_to_finetuning_2026.ipynb](finetuning/01_intro_to_finetuning_2026.ipynb) | FT | When to finetune vs prompt vs retrieve — the decision rubric |
| 6 | [multimodal/01_intro_to_multimodal_systems.ipynb](multimodal/01_intro_to_multimodal_systems.ipynb) | MM | Where modalities enter the stack, perception budgets |
| 7 | [systems/01_intro_to_llm_systems_2026.ipynb](systems/01_intro_to_llm_systems_2026.ipynb) | SY | Why the runtime matters as much as the model |

**Checkpoint**: You can now explain to someone *why* there are 7 disciplines, how they relate, and roughly when each one is the right tool.

---

## Orbit 1 — Prompt Craft & First Evals

**Goal**: Master the interface to LLMs (prompting) and immediately learn to *measure* what you build. These two skills are inseparable.

### 1A: Prompt Foundations

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [prompt-engineering/basic-prompt-structures.ipynb](prompt-engineering/basic-prompt-structures.ipynb) | PE |
| 2 | [prompt-engineering/prompt-templates-variables-jinja2.ipynb](prompt-engineering/prompt-templates-variables-jinja2.ipynb) | PE |
| 3 | [prompt-engineering/zero-shot-prompting.ipynb](prompt-engineering/zero-shot-prompting.ipynb) | PE |
| 4 | [prompt-engineering/few-shot-learning.ipynb](prompt-engineering/few-shot-learning.ipynb) | PE |
| 5 | [prompt-engineering/cot-prompting.ipynb](prompt-engineering/cot-prompting.ipynb) | PE |
| 6 | [prompt-engineering/role-prompting.ipynb](prompt-engineering/role-prompting.ipynb) | PE |
| 7 | [prompt-engineering/instruction-engineering-notebook.ipynb](prompt-engineering/instruction-engineering-notebook.ipynb) | PE |

### 1B: Measuring What You Prompt (Evals Cross-Pollination)

| # | Notebook | Domain |
|---|----------|--------|
| 8 | [evals/02_building_eval_datasets.ipynb](evals/02_building_eval_datasets.ipynb) | EV |
| 9 | [evals/03_metrics_from_scratch.ipynb](evals/03_metrics_from_scratch.ipynb) | EV |
| 10 | [evals/05_rule_based_prompt_eval.ipynb](evals/05_rule_based_prompt_eval.ipynb) | EV |
| 11 | [prompt-engineering/evaluating-prompt-effectiveness.ipynb](prompt-engineering/evaluating-prompt-effectiveness.ipynb) | PE |
| 12 | [evals/06_rubrics_and_llm_as_judge.ipynb](evals/06_rubrics_and_llm_as_judge.ipynb) | EV |

### 1C: Advanced Prompt Strategies

| # | Notebook | Domain |
|---|----------|--------|
| 13 | [prompt-engineering/self-consistency.ipynb](prompt-engineering/self-consistency.ipynb) | PE |
| 14 | [prompt-engineering/constrained-guided-generation.ipynb](prompt-engineering/constrained-guided-generation.ipynb) | PE |
| 15 | [prompt-engineering/task-decomposition-prompts.ipynb](prompt-engineering/task-decomposition-prompts.ipynb) | PE |
| 16 | [prompt-engineering/prompt-chaining-sequencing.ipynb](prompt-engineering/prompt-chaining-sequencing.ipynb) | PE |
| 17 | [prompt-engineering/negative-prompting.ipynb](prompt-engineering/negative-prompting.ipynb) | PE |
| 18 | [prompt-engineering/ambiguity-clarity.ipynb](prompt-engineering/ambiguity-clarity.ipynb) | PE |
| 19 | [prompt-engineering/prompt-length-complexity-management.ipynb](prompt-engineering/prompt-length-complexity-management.ipynb) | PE |
| 20 | [prompt-engineering/prompt-formatting-structure.ipynb](prompt-engineering/prompt-formatting-structure.ipynb) | PE |
| 21 | [prompt-engineering/specific-task-prompts.ipynb](prompt-engineering/specific-task-prompts.ipynb) | PE |
| 22 | [prompt-engineering/prompt-optimization-techniques.ipynb](prompt-engineering/prompt-optimization-techniques.ipynb) | PE |

### 1D: Safety, Ethics & Experiment Design

| # | Notebook | Domain |
|---|----------|--------|
| 23 | [prompt-engineering/prompt-security-and-safety.ipynb](prompt-engineering/prompt-security-and-safety.ipynb) | PE |
| 24 | [prompt-engineering/ethical-prompt-engineering.ipynb](prompt-engineering/ethical-prompt-engineering.ipynb) | PE |
| 25 | [prompt-engineering/multilingual-prompting.ipynb](prompt-engineering/multilingual-prompting.ipynb) | PE |
| 26 | [evals/07_pairwise_and_preference_eval.ipynb](evals/07_pairwise_and_preference_eval.ipynb) | EV |
| 27 | [evals/08_prompt_experiment_design.ipynb](evals/08_prompt_experiment_design.ipynb) | EV |
| 28 | [evals/04_error_analysis_and_failure_buckets.ipynb](evals/04_error_analysis_and_failure_buckets.ipynb) | EV |

**Checkpoint**: You can design prompts, template them, evaluate them with metrics and judges, and run A/B experiments. You *never prompt blind* from this point forward.

---

## Orbit 2 — Retrieval: Giving LLMs Knowledge

**Goal**: Build retrieval systems of increasing sophistication, and *always evaluate them*. Interleave RAG techniques with RAG-specific evals.

### 2A: RAG Fundamentals

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [rag/simple_rag.ipynb](rag/simple_rag.ipynb) (revisit architecture) | RAG |
| 2 | [rag/simple_csv_rag.ipynb](rag/simple_csv_rag.ipynb) | RAG |
| 3 | [rag/reliable_rag.ipynb](rag/reliable_rag.ipynb) | RAG |
| 4 | [rag/choose_chunk_size.ipynb](rag/choose_chunk_size.ipynb) | RAG |
| 5 | [rag/proposition_chunking.ipynb](rag/proposition_chunking.ipynb) | RAG |
| 6 | [rag/semantic_chunking.ipynb](rag/semantic_chunking.ipynb) | RAG |

### 2B: Query & Context Enhancement

| # | Notebook | Domain |
|---|----------|--------|
| 7 | [rag/query_transformations.ipynb](rag/query_transformations.ipynb) | RAG |
| 8 | [rag/HyDe_Hypothetical_Document_Embedding.ipynb](rag/HyDe_Hypothetical_Document_Embedding.ipynb) | RAG |
| 9 | [rag/HyPE_Hypothetical_Prompt_Embeddings.ipynb](rag/HyPE_Hypothetical_Prompt_Embeddings.ipynb) | RAG |
| 10 | [rag/contextual_chunk_headers.ipynb](rag/contextual_chunk_headers.ipynb) | RAG |
| 11 | [rag/relevant_segment_extraction.ipynb](rag/relevant_segment_extraction.ipynb) | RAG |
| 12 | [rag/context_enrichment_window_around_chunk.ipynb](rag/context_enrichment_window_around_chunk.ipynb) | RAG |
| 13 | [rag/contextual_compression.ipynb](rag/contextual_compression.ipynb) | RAG |
| 14 | [rag/document_augmentation.ipynb](rag/document_augmentation.ipynb) | RAG |

### 2C: RAG Evals (Cross-Pollination)

| # | Notebook | Domain |
|---|----------|--------|
| 15 | [evals/09_retrieval_metrics.ipynb](evals/09_retrieval_metrics.ipynb) | EV |
| 16 | [evals/10_faithfulness_and_groundedness.ipynb](evals/10_faithfulness_and_groundedness.ipynb) | EV |
| 17 | [evals/11_citation_and_evidence_coverage.ipynb](evals/11_citation_and_evidence_coverage.ipynb) | EV |
| 18 | [rag/explainable_retrieval.ipynb](rag/explainable_retrieval.ipynb) | RAG |

### 2D: Advanced Retrieval Architectures

| # | Notebook | Domain |
|---|----------|--------|
| 19 | [rag/fusion_retrieval.ipynb](rag/fusion_retrieval.ipynb) | RAG |
| 20 | [rag/reranking.ipynb](rag/reranking.ipynb) | RAG |
| 21 | [rag/hierarchical_indices.ipynb](rag/hierarchical_indices.ipynb) | RAG |
| 22 | [rag/dartboard.ipynb](rag/dartboard.ipynb) | RAG |
| 23 | [rag/adaptive_retrieval.ipynb](rag/adaptive_retrieval.ipynb) | RAG |
| 24 | [rag/retrieval_with_feedback_loop.ipynb](rag/retrieval_with_feedback_loop.ipynb) | RAG |

### 2E: Graph & Structured RAG

| # | Notebook | Domain |
|---|----------|--------|
| 25 | [rag/graph_rag.ipynb](rag/graph_rag.ipynb) | RAG |
| 26 | [rag/Microsoft_GraphRag.ipynb](rag/Microsoft_GraphRag.ipynb) | RAG |
| 27 | [rag/raptor.ipynb](rag/raptor.ipynb) | RAG |
| 28 | [rag/self_rag.ipynb](rag/self_rag.ipynb) | RAG |
| 29 | [rag/crag.ipynb](rag/crag.ipynb) | RAG |
| 30 | [evals/12_rag_ablation_lab.ipynb](evals/12_rag_ablation_lab.ipynb) | EV |

**Checkpoint**: You can build, tune, and rigorously evaluate retrieval systems. You understand chunking, enhancement, reranking, graphs, and self-correcting retrieval — and you can *prove* which approach works best with metrics.

---

## Orbit 3 — Agents: Systems That Think and Act

**Goal**: Build agents from scratch, starting from the loop and scaling to multi-agent systems. Weave in tool design, memory, and agent-specific evals.

### 3A: Agent Foundations

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [agents/02_the_agent_loop.ipynb](agents/02_the_agent_loop.ipynb) | AG |
| 2 | [agents/03_tool_use_and_function_calling.ipynb](agents/03_tool_use_and_function_calling.ipynb) | AG |
| 3 | [agents/04_structured_output_parsing.ipynb](agents/04_structured_output_parsing.ipynb) | AG |
| 4 | [agents/05_building_a_react_agent.ipynb](agents/05_building_a_react_agent.ipynb) | AG |

### 3B: Single-Agent Reasoning Patterns

| # | Notebook | Domain |
|---|----------|--------|
| 5 | [agents/06_plan_and_execute.ipynb](agents/06_plan_and_execute.ipynb) | AG |
| 6 | [agents/07_reflection_and_self_critique.ipynb](agents/07_reflection_and_self_critique.ipynb) | AG |
| 7 | [agents/08_tree_of_thought.ipynb](agents/08_tree_of_thought.ipynb) | AG |
| 8 | [agents/09_iterative_refinement.ipynb](agents/09_iterative_refinement.ipynb) | AG |

### 3C: Agent Memory (connects back to RAG)

| # | Notebook | Domain |
|---|----------|--------|
| 9 | [agents/10_agent_memory_short_term.ipynb](agents/10_agent_memory_short_term.ipynb) | AG |
| 10 | [agents/11_agent_memory_long_term.ipynb](agents/11_agent_memory_long_term.ipynb) | AG |
| 11 | [agents/12_knowledge_graph_memory.ipynb](agents/12_knowledge_graph_memory.ipynb) | AG |

### 3D: Tool Engineering

| # | Notebook | Domain |
|---|----------|--------|
| 12 | [agents/13_advanced_tool_design.ipynb](agents/13_advanced_tool_design.ipynb) | AG |
| 13 | [agents/14_code_execution_tool.ipynb](agents/14_code_execution_tool.ipynb) | AG |
| 14 | [agents/15_web_and_search_tools.ipynb](agents/15_web_and_search_tools.ipynb) | AG |
| 15 | [agents/16_file_and_data_tools.ipynb](agents/16_file_and_data_tools.ipynb) | AG |

### 3E: Agent Evals (Cross-Pollination)

| # | Notebook | Domain |
|---|----------|--------|
| 16 | [evals/13_tool_use_and_task_success_eval.ipynb](evals/13_tool_use_and_task_success_eval.ipynb) | EV |
| 17 | [evals/14_trajectory_grading.ipynb](evals/14_trajectory_grading.ipynb) | EV |

### 3F: Multi-Agent Systems

| # | Notebook | Domain |
|---|----------|--------|
| 18 | [agents/17_multi_agent_conversation.ipynb](agents/17_multi_agent_conversation.ipynb) | AG |
| 19 | [agents/18_agent_debate_and_consensus.ipynb](agents/18_agent_debate_and_consensus.ipynb) | AG |
| 20 | [agents/19_sequential_agent_pipelines.ipynb](agents/19_sequential_agent_pipelines.ipynb) | AG |
| 21 | [agents/20_hierarchical_agent_delegation.ipynb](agents/20_hierarchical_agent_delegation.ipynb) | AG |
| 22 | [agents/21_agent_orchestration_patterns.ipynb](agents/21_agent_orchestration_patterns.ipynb) | AG |
| 23 | [agents/22_shared_state_and_blackboard.ipynb](agents/22_shared_state_and_blackboard.ipynb) | AG |
| 24 | [agents/23_swarm_intelligence.ipynb](agents/23_swarm_intelligence.ipynb) | AG |

### 3G: Multi-Agent & Robustness Evals

| # | Notebook | Domain |
|---|----------|--------|
| 25 | [evals/15_multi_agent_system_eval.ipynb](evals/15_multi_agent_system_eval.ipynb) | EV |
| 26 | [evals/17_agent_robustness_and_adversarial_eval.ipynb](evals/17_agent_robustness_and_adversarial_eval.ipynb) | EV |

### 3H: Production Concerns

| # | Notebook | Domain |
|---|----------|--------|
| 27 | [agents/24_agent_safety_and_guardrails.ipynb](agents/24_agent_safety_and_guardrails.ipynb) | AG |
| 28 | [agents/25_human_in_the_loop.ipynb](agents/25_human_in_the_loop.ipynb) | AG |
| 29 | [agents/26_agent_evaluation_and_testing.ipynb](agents/26_agent_evaluation_and_testing.ipynb) | AG |
| 30 | [agents/27_cost_and_latency_optimization.ipynb](agents/27_cost_and_latency_optimization.ipynb) | AG |
| 31 | [agents/28_error_handling_and_resilience.ipynb](agents/28_error_handling_and_resilience.ipynb) | AG |
| 32 | [evals/16_cost_latency_reliability_eval.ipynb](evals/16_cost_latency_reliability_eval.ipynb) | EV |

### 3I: Agentic RAG (Convergence of RAG + Agents)

| # | Notebook | Domain |
|---|----------|--------|
| 33 | [rag/Agentic_RAG.ipynb](rag/Agentic_RAG.ipynb) | RAG |
| 34 | [rag/multi_model_rag_with_captioning.ipynb](rag/multi_model_rag_with_captioning.ipynb) | RAG |
| 35 | [rag/multi_model_rag_with_colpali.ipynb](rag/multi_model_rag_with_colpali.ipynb) | RAG |

### 3J: Communication Protocols

| # | Notebook | Domain |
|---|----------|--------|
| 36 | [agents/29_mcp_from_scratch.ipynb](agents/29_mcp_from_scratch.ipynb) | AG |
| 37 | [agents/30_a2a_protocol.ipynb](agents/30_a2a_protocol.ipynb) | AG |
| 38 | [agents/31_building_an_agent_runtime.ipynb](agents/31_building_an_agent_runtime.ipynb) | AG |

**Checkpoint**: You can build single and multi-agent systems, equip them with tools and memory, evaluate their trajectories, and harden them for production. You understand MCP, A2A, and agent runtimes.

---

## Orbit 4 — Multimodal: Beyond Text

**Goal**: Extend everything you've learned to vision, audio, and video. Build multimodal RAG and multimodal agents.

### 4A: Multimodal Foundations & Benchmarking

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [multimodal/02_model_families_and_modality_interfaces.ipynb](multimodal/02_model_families_and_modality_interfaces.ipynb) | MM |
| 2 | [multimodal/03_patches_tokens_spectrograms_and_budgeting.ipynb](multimodal/03_patches_tokens_spectrograms_and_budgeting.ipynb) | MM |
| 3 | [multimodal/04_building_a_multimodal_benchmark_harness.ipynb](multimodal/04_building_a_multimodal_benchmark_harness.ipynb) | MM |

### 4B: Vision & Document Understanding

| # | Notebook | Domain |
|---|----------|--------|
| 4 | [multimodal/05_image_prompting_and_visual_reasoning.ipynb](multimodal/05_image_prompting_and_visual_reasoning.ipynb) | MM |
| 5 | [multimodal/06_ocr_layout_and_table_extraction.ipynb](multimodal/06_ocr_layout_and_table_extraction.ipynb) | MM |
| 6 | [multimodal/07_structured_outputs_grounding_and_localization.ipynb](multimodal/07_structured_outputs_grounding_and_localization.ipynb) | MM |
| 7 | [multimodal/08_small_vlms_and_multi_image_workflows.ipynb](multimodal/08_small_vlms_and_multi_image_workflows.ipynb) | MM |

### 4C: Multimodal RAG (Convergence of RAG + Multimodal)

| # | Notebook | Domain |
|---|----------|--------|
| 8 | [multimodal/09_captioning_as_a_text_bridge.ipynb](multimodal/09_captioning_as_a_text_bridge.ipynb) | MM |
| 9 | [multimodal/10_page_as_image_retrieval_with_colpali.ipynb](multimodal/10_page_as_image_retrieval_with_colpali.ipynb) | MM |
| 10 | [multimodal/11_hybrid_text_plus_vision_retrieval.ipynb](multimodal/11_hybrid_text_plus_vision_retrieval.ipynb) | MM |
| 11 | [multimodal/12_multimodal_grounding_and_evaluation.ipynb](multimodal/12_multimodal_grounding_and_evaluation.ipynb) | MM |

### 4D: Audio & Speech

| # | Notebook | Domain |
|---|----------|--------|
| 12 | [multimodal/13_speech_recognition_and_transcription.ipynb](multimodal/13_speech_recognition_and_transcription.ipynb) | MM |
| 13 | [multimodal/14_audio_understanding_classification_and_tagging.ipynb](multimodal/14_audio_understanding_classification_and_tagging.ipynb) | MM |
| 14 | [multimodal/15_speech_plus_document_workflows.ipynb](multimodal/15_speech_plus_document_workflows.ipynb) | MM |
| 15 | [multimodal/16_multimodal_agents_across_speech_and_vision.ipynb](multimodal/16_multimodal_agents_across_speech_and_vision.ipynb) | MM |

### 4E: Video & Operations

| # | Notebook | Domain |
|---|----------|--------|
| 16 | [multimodal/17_video_understanding_frame_sampling_and_temporal_reasoning.ipynb](multimodal/17_video_understanding_frame_sampling_and_temporal_reasoning.ipynb) | MM |
| 17 | [multimodal/18_video_grounding_summarization_and_event_extraction.ipynb](multimodal/18_video_grounding_summarization_and_event_extraction.ipynb) | MM |
| 18 | [multimodal/19_serving_cost_safety_and_evaluation_for_multimodal.ipynb](multimodal/19_serving_cost_safety_and_evaluation_for_multimodal.ipynb) | MM |

**Checkpoint**: You can work across modalities — images, documents, audio, video — and build multimodal RAG pipelines and agents. You understand perception budgets and multimodal evaluation.

---

## Orbit 5 — Finetuning: Changing the Model Itself

**Goal**: Only now, after mastering prompting, RAG, agents, and evals, do you learn to *change the model*. This sequence is intentional — finetuning is the last lever, not the first.

### 5A: Foundations & Environment

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [finetuning/02_colab_pro_and_unsloth_setup.ipynb](finetuning/02_colab_pro_and_unsloth_setup.ipynb) | FT |
| 2 | [finetuning/03_model_selection_and_vram_budgeting.ipynb](finetuning/03_model_selection_and_vram_budgeting.ipynb) | FT |
| 3 | [finetuning/04_datasets_chat_templates_and_loss_masking.ipynb](finetuning/04_datasets_chat_templates_and_loss_masking.ipynb) | FT |

### 5B: Supervised Fine-Tuning

| # | Notebook | Domain |
|---|----------|--------|
| 4 | [finetuning/05_first_qlora_sft_run.ipynb](finetuning/05_first_qlora_sft_run.ipynb) | FT |
| 5 | [finetuning/06_data_curation_cleaning_and_splitting.ipynb](finetuning/06_data_curation_cleaning_and_splitting.ipynb) | FT |
| 6 | [finetuning/07_lora_hyperparameters_and_target_modules.ipynb](finetuning/07_lora_hyperparameters_and_target_modules.ipynb) | FT |
| 7 | [finetuning/08_packing_long_context_and_throughput.ipynb](finetuning/08_packing_long_context_and_throughput.ipynb) | FT |
| 8 | [finetuning/09_adapter_inspection_merging_and_export.ipynb](finetuning/09_adapter_inspection_merging_and_export.ipynb) | FT |

### 5C: Preference Alignment

| # | Notebook | Domain |
|---|----------|--------|
| 9 | [finetuning/10_preference_data_construction.ipynb](finetuning/10_preference_data_construction.ipynb) | FT |
| 10 | [finetuning/11_dpo_alignment.ipynb](finetuning/11_dpo_alignment.ipynb) | FT |
| 11 | [finetuning/12_orpo_and_kto_alignment.ipynb](finetuning/12_orpo_and_kto_alignment.ipynb) | FT |
| 12 | [finetuning/13_finetuning_evaluation_and_regressions.ipynb](finetuning/13_finetuning_evaluation_and_regressions.ipynb) | FT |

### 5D: Domain Adaptation & Data Scaling

| # | Notebook | Domain |
|---|----------|--------|
| 13 | [finetuning/14_continued_pretraining.ipynb](finetuning/14_continued_pretraining.ipynb) | FT |
| 14 | [finetuning/15_synthetic_data_and_distillation.ipynb](finetuning/15_synthetic_data_and_distillation.ipynb) | FT |
| 15 | [finetuning/16_forgetting_mixture_design_and_safety.ipynb](finetuning/16_forgetting_mixture_design_and_safety.ipynb) | FT |

### 5E: RL & Reasoning Post-Training

| # | Notebook | Domain |
|---|----------|--------|
| 16 | [finetuning/17_grpo_foundations_and_reward_design.ipynb](finetuning/17_grpo_foundations_and_reward_design.ipynb) | FT |
| 17 | [finetuning/18_reasoning_finetuning_with_grpo.ipynb](finetuning/18_reasoning_finetuning_with_grpo.ipynb) | FT |
| 18 | [finetuning/19_rl_for_tool_use_and_structured_outputs.ipynb](finetuning/19_rl_for_tool_use_and_structured_outputs.ipynb) | FT |

**Checkpoint**: You can finetune models with SFT, align them with DPO/ORPO/KTO, and push into RL territory. Critically, you know when *not* to finetune because you've already mastered the cheaper levers.

---

## Orbit 6 — Systems: Serving, Scaling & Operating

**Goal**: Deploy everything you've built. Understand inference runtimes, quantization, serving, routing, and operational reliability.

### 6A: Measurement & Local Inference

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [systems/02_runtime_landscape_and_deployment_tiers.ipynb](systems/02_runtime_landscape_and_deployment_tiers.ipynb) | SY |
| 2 | [systems/03_latency_throughput_and_memory_budgeting.ipynb](systems/03_latency_throughput_and_memory_budgeting.ipynb) | SY |
| 3 | [systems/04_building_a_reproducible_benchmark_harness.ipynb](systems/04_building_a_reproducible_benchmark_harness.ipynb) | SY |
| 4 | [systems/05_gguf_and_llama_cpp_basics.ipynb](systems/05_gguf_and_llama_cpp_basics.ipynb) | SY |
| 5 | [systems/06_quantization_and_memory_tradeoffs.ipynb](systems/06_quantization_and_memory_tradeoffs.ipynb) | SY |
| 6 | [systems/07_local_openai_compatible_serving.ipynb](systems/07_local_openai_compatible_serving.ipynb) | SY |

### 6B: Production Serving Runtimes

| # | Notebook | Domain |
|---|----------|--------|
| 7 | [systems/08_transformers_baselines_and_batching.ipynb](systems/08_transformers_baselines_and_batching.ipynb) | SY |
| 8 | [systems/09_vllm_quickstart_and_api_serving.ipynb](systems/09_vllm_quickstart_and_api_serving.ipynb) | SY |
| 9 | [systems/10_continuous_batching_and_paged_kv_cache.ipynb](systems/10_continuous_batching_and_paged_kv_cache.ipynb) | SY |
| 10 | [systems/11_prefix_caching_chunked_prefill_and_long_context.ipynb](systems/11_prefix_caching_chunked_prefill_and_long_context.ipynb) | SY |
| 11 | [systems/12_structured_outputs_tool_calls_and_response_contracts.ipynb](systems/12_structured_outputs_tool_calls_and_response_contracts.ipynb) | SY |
| 12 | [systems/13_multi_lora_serving_and_runtime_model_selection.ipynb](systems/13_multi_lora_serving_and_runtime_model_selection.ipynb) | SY |

### 6C: Advanced Patterns & Distributed Inference

| # | Notebook | Domain |
|---|----------|--------|
| 13 | [systems/14_sglang_and_advanced_scheduling_patterns.ipynb](systems/14_sglang_and_advanced_scheduling_patterns.ipynb) | SY |
| 14 | [systems/15_speculative_decoding_and_assisted_generation.ipynb](systems/15_speculative_decoding_and_assisted_generation.ipynb) | SY |
| 15 | [systems/16_routing_retries_admission_control_and_load_shedding.ipynb](systems/16_routing_retries_admission_control_and_load_shedding.ipynb) | SY |
| 16 | [systems/17_distributed_inference_concepts.ipynb](systems/17_distributed_inference_concepts.ipynb) | SY |

### 6D: Operations & Reliability (Convergence with Evals)

| # | Notebook | Domain |
|---|----------|--------|
| 17 | [systems/18_observability_metrics_tracing_and_cost_accounting.ipynb](systems/18_observability_metrics_tracing_and_cost_accounting.ipynb) | SY |
| 18 | [systems/19_eval_driven_rollouts_regressions_and_incident_playbooks.ipynb](systems/19_eval_driven_rollouts_regressions_and_incident_playbooks.ipynb) | SY |
| 19 | [evals/18_regression_testing_for_llm_systems.ipynb](evals/18_regression_testing_for_llm_systems.ipynb) | EV |
| 20 | [evals/19_safety_and_policy_evals.ipynb](evals/19_safety_and_policy_evals.ipynb) | EV |
| 21 | [evals/20_human_eval_workflows.ipynb](evals/20_human_eval_workflows.ipynb) | EV |
| 22 | [evals/21_experiment_tracking_and_reporting.ipynb](evals/21_experiment_tracking_and_reporting.ipynb) | EV |

**Checkpoint**: You can serve models locally and at scale, understand quantization, batching, caching, routing, and distributed inference. You know how to operate LLM systems with observability, regression gates, and incident playbooks.

---

## Orbit 7 — Capstone Convergence: All Disciplines Together

**Goal**: Build complete end-to-end systems that combine agents, RAG, multimodal, finetuning, systems, and evals. Every capstone project is a convergence point.

### 7A: Castalia Scholar (Agents Capstone)

| # | Notebook | Domain |
|---|----------|--------|
| 1 | [agents/32_project_architecture.ipynb](agents/32_project_architecture.ipynb) | AG |
| 2 | [agents/33_project_retrieval_agent.ipynb](agents/33_project_retrieval_agent.ipynb) | AG |
| 3 | [agents/34_project_analysis_agent.ipynb](agents/34_project_analysis_agent.ipynb) | AG |
| 4 | [agents/35_project_writing_agent.ipynb](agents/35_project_writing_agent.ipynb) | AG |
| 5 | [agents/36_project_review_agent.ipynb](agents/36_project_review_agent.ipynb) | AG |
| 6 | [agents/37_project_full_system.ipynb](agents/37_project_full_system.ipynb) | AG |

### 7B: Castalia Perception (Multimodal Capstone)

| # | Notebook | Domain |
|---|----------|--------|
| 7 | [multimodal/20_project_architecture_and_dataset.ipynb](multimodal/20_project_architecture_and_dataset.ipynb) | MM |
| 8 | [multimodal/21_project_pipeline_and_benchmark.ipynb](multimodal/21_project_pipeline_and_benchmark.ipynb) | MM |
| 9 | [multimodal/22_project_handoff_and_evaluation_report.ipynb](multimodal/22_project_handoff_and_evaluation_report.ipynb) | MM |

### 7C: Castalia Mentor (Finetuning Capstone)

| # | Notebook | Domain |
|---|----------|--------|
| 10 | [finetuning/20_project_dataset_pipeline.ipynb](finetuning/20_project_dataset_pipeline.ipynb) | FT |
| 11 | [finetuning/21_project_training_pipeline.ipynb](finetuning/21_project_training_pipeline.ipynb) | FT |
| 12 | [finetuning/22_project_benchmark_and_export.ipynb](finetuning/22_project_benchmark_and_export.ipynb) | FT |

### 7D: Castalia Runtime (Systems Capstone)

| # | Notebook | Domain |
|---|----------|--------|
| 13 | [systems/20_project_runtime_prototype.ipynb](systems/20_project_runtime_prototype.ipynb) | SY |
| 14 | [systems/21_project_benchmark_routing_and_structured_outputs.ipynb](systems/21_project_benchmark_routing_and_structured_outputs.ipynb) | SY |
| 15 | [systems/22_project_deployment_handoff.ipynb](systems/22_project_deployment_handoff.ipynb) | SY |

### 7E: Castalia Bench (Evals Capstone — The Grand Finale)

| # | Notebook | Domain |
|---|----------|--------|
| 16 | [evals/22_castalia_bench_capstone.ipynb](evals/22_castalia_bench_capstone.ipynb) | EV |

**Checkpoint**: You've built a multi-agent research assistant, a multimodal perception system, a finetuned mentor model, a production runtime, and a comprehensive benchmark suite — all interconnected.

---

## Cross-Cutting Themes Map

These themes thread through multiple orbits. Use this to find *all* notebooks related to a concept:

### 🔒 Safety & Security
- [prompt-security-and-safety.ipynb](prompt-security-and-safety.ipynb) (Orbit 1)
- [ethical-prompt-engineering.ipynb](ethical-prompt-engineering.ipynb) (Orbit 1)
- [agents/24_agent_safety_and_guardrails.ipynb](agents/24_agent_safety_and_guardrails.ipynb) (Orbit 3)
- [agents/25_human_in_the_loop.ipynb](agents/25_human_in_the_loop.ipynb) (Orbit 3)
- [evals/19_safety_and_policy_evals.ipynb](evals/19_safety_and_policy_evals.ipynb) (Orbit 6)
- [evals/17_agent_robustness_and_adversarial_eval.ipynb](evals/17_agent_robustness_and_adversarial_eval.ipynb) (Orbit 3)
- [finetuning/16_forgetting_mixture_design_and_safety.ipynb](finetuning/16_forgetting_mixture_design_and_safety.ipynb) (Orbit 5)
- [multimodal/19_serving_cost_safety_and_evaluation_for_multimodal.ipynb](multimodal/19_serving_cost_safety_and_evaluation_for_multimodal.ipynb) (Orbit 4)

### 📊 Evaluation (everywhere)
- Evals notebooks are distributed across Orbits 1, 2, 3, 6, and 7 — never isolated
- Agent-internal evals: [agents/26_agent_evaluation_and_testing.ipynb](agents/26_agent_evaluation_and_testing.ipynb)
- Finetuning evals: [finetuning/13_finetuning_evaluation_and_regressions.ipynb](finetuning/13_finetuning_evaluation_and_regressions.ipynb)
- Multimodal evals: [multimodal/12_multimodal_grounding_and_evaluation.ipynb](multimodal/12_multimodal_grounding_and_evaluation.ipynb)

### 💰 Cost, Latency & Optimization
- [agents/27_cost_and_latency_optimization.ipynb](agents/27_cost_and_latency_optimization.ipynb) (Orbit 3)
- [evals/16_cost_latency_reliability_eval.ipynb](evals/16_cost_latency_reliability_eval.ipynb) (Orbit 3)
- [systems/03_latency_throughput_and_memory_budgeting.ipynb](systems/03_latency_throughput_and_memory_budgeting.ipynb) (Orbit 6)
- [systems/18_observability_metrics_tracing_and_cost_accounting.ipynb](systems/18_observability_metrics_tracing_and_cost_accounting.ipynb) (Orbit 6)
- [multimodal/03_patches_tokens_spectrograms_and_budgeting.ipynb](multimodal/03_patches_tokens_spectrograms_and_budgeting.ipynb) (Orbit 4)

### 🔧 Structured Output & Tool Contracts
- [constrained-guided-generation.ipynb](constrained-guided-generation.ipynb) (Orbit 1)
- [agents/04_structured_output_parsing.ipynb](agents/04_structured_output_parsing.ipynb) (Orbit 3)
- [agents/13_advanced_tool_design.ipynb](agents/13_advanced_tool_design.ipynb) (Orbit 3)
- [systems/12_structured_outputs_tool_calls_and_response_contracts.ipynb](systems/12_structured_outputs_tool_calls_and_response_contracts.ipynb) (Orbit 6)
- [finetuning/19_rl_for_tool_use_and_structured_outputs.ipynb](finetuning/19_rl_for_tool_use_and_structured_outputs.ipynb) (Orbit 5)

### 🧠 Memory & Knowledge
- `agents/10–12` short-term, long-term, knowledge graph memory (Orbit 3)
- [rag/graph_rag.ipynb](rag/graph_rag.ipynb), [rag/Microsoft_GraphRag.ipynb](rag/Microsoft_GraphRag.ipynb) (Orbit 2)
- [agents/22_shared_state_and_blackboard.ipynb](agents/22_shared_state_and_blackboard.ipynb) (Orbit 3)

### 🌐 Protocols & Interoperability
- [agents/29_mcp_from_scratch.ipynb](agents/29_mcp_from_scratch.ipynb) (Orbit 3)
- [agents/30_a2a_protocol.ipynb](agents/30_a2a_protocol.ipynb) (Orbit 3)
- [agents/31_building_an_agent_runtime.ipynb](agents/31_building_an_agent_runtime.ipynb) (Orbit 3)
- [systems/07_local_openai_compatible_serving.ipynb](systems/07_local_openai_compatible_serving.ipynb) (Orbit 6)

---

## Dependency Graph (Orbits)

```
                    ┌─────────────┐
                    │  Orbit 0    │
                    │ Ground Zero │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Orbit 1    │
                    │ Prompt+Eval │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  Orbit 2    │    │     │  Orbit 3    │
       │    RAG      │◄───┼────►│   Agents    │
       └──────┬──────┘    │     └──────┬──────┘
              │            │            │
              └────────┬───┘────────────┘
                       │
              ┌────────▼────────┐
              │    Orbit 4      │
              │   Multimodal    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Orbit 5      │
              │  Finetuning     │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Orbit 6      │
              │    Systems      │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Orbit 7      │
              │   Capstones     │
              └─────────────────┘
```

**Key**: Orbits 2 and 3 can be done in parallel or interleaved. Orbit 4 requires Orbits 2+3. Orbit 5 requires Orbits 1–4 (you need to know cheaper levers first). Orbit 6 benefits from all prior orbits. Orbit 7 requires everything.

---

## Notes & Considerations

1. **Evals are never a standalone phase** — they are woven into every orbit because measurement must be habitual, not an afterthought.

2. **The "last lever" principle for finetuning** — the finetuning intro notebook itself says: "change the model only after cheaper levers have been used well and measured honestly." The orbit ordering respects this philosophy.

3. **Orbits 2 and 3 are intentionally parallel-ready** — a learner can alternate between RAG and Agents week by week, since agent memory (Orbit 3C) explicitly builds on retrieval concepts.

4. **Each orbit ends with a checkpoint** — use these as self-assessment points. If you can't explain the checkpoint statement confidently, revisit the orbit.

5. **Cross-cutting themes are your review tool** — when you reach Orbit 5 (finetuning for tool use), go back and review the tool design notebooks from Orbit 3. The theme maps make this easy.

6. **Total estimated notebook count**: ~176 notebooks. This is a comprehensive curriculum — pace yourself and use the orbits as natural break points.
