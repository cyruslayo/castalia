# Multimodal - Vision, Document Intelligence, Audio, and Video Systems

**A notebook-based course on modern open-source multimodal systems, retrieval, evaluation, and deployment for 2026.**

This folder contains 22 Jupyter notebooks organized into 6 modules, progressing from multimodal foundations to a full capstone that combines image, document, audio, and video workflows. The focus is not generic media AI theory. It is the practical engineering layer that begins once text-only systems are no longer enough: how to reason over images and pages, transcribe and analyze audio, understand short videos, benchmark multimodal outputs, and ship open-source workflows with clear contracts and measurable trade-offs.

| Component | Implementation |
|---|---|
| **General VLMs** | Qwen2.5-VL + SmolVLM |
| **Document Retrieval** | ColPali + hybrid text/vision retrieval |
| **Audio Stack** | Whisper + Qwen2-Audio + CLAP |
| **Video Utilities** | Qwen2.5-VL video inputs + frame sampling with `decord` |
| **Artifacts** | JSON/CSV reports, grounded outputs, evaluation tables, benchmark traces |
| **Framework** | Notebook-first, open-source only, minimal abstractions |

> **Prerequisites:** Complete [prompt-engineering/](../prompt-engineering/), [rag/](../rag/), [agents/](../agents/), [evals/](../evals/), [finetuning/](../finetuning/), and [systems/](../systems/) first. This track assumes you already know how to control models, ground them, orchestrate them, evaluate them, adapt them, and run them as systems.
> **Runtime:** Colab-friendly where practical, with explicit honesty about heavier workloads. Image, document, and audio notebooks are designed to run comfortably on T4/L4-class environments; video notebooks use bounded clips and sampled frames rather than pretending every learner has a large multi-GPU setup.

---

## Why This Track Exists

The earlier Castalia tracks teach how to:
- control models with prompts
- ground them with retrieval
- orchestrate them with tools and multi-agent loops
- measure quality and regressions
- adapt model behavior through finetuning
- run open models as serving systems

This track teaches what comes next: how to build systems that can **perceive and reason across modalities** instead of treating the world as plain text.

In other words:
- `prompt-engineering/` teaches **control**
- `rag/` teaches **grounding**
- `agents/` teaches **orchestration**
- `evals/` teaches **measurement**
- `finetuning/` teaches **behavior change**
- `systems/` teaches **runtime engineering**
- `multimodal/` teaches **perception, grounding, and multimodal operations**

That shift matters. Many real workloads are not text-first:
- invoices, forms, and reports are page-layout problems
- screenshots and dashboards are image-grounding problems
- meetings and voice notes are speech problems
- tutorials, inspections, and user recordings are short-video problems

If the interface stays text-only, those signals are either lost or awkwardly approximated. Multimodal systems close that gap.

---

## Course Structure

### Module 1: Foundations and Interfaces (Notebooks 01-04)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_intro_to_multimodal_systems.ipynb` | Why multimodal work starts after text-only systems, where modalities enter the stack, and how to think about perception budgets |
| 02 | `02_model_families_and_modality_interfaces.ipynb` | The current open model landscape for image, document, audio, and video tasks, plus common message and processor interfaces |
| 03 | `03_patches_tokens_spectrograms_and_budgeting.ipynb` | Image patches, audio windows, sampled frames, and the cost/latency reasoning needed before choosing a model |
| 04 | `04_building_a_multimodal_benchmark_harness.ipynb` | From-scratch benchmark design for multimodal prompts, retrieval, extraction, and grounding tasks |

### Module 2: Vision and Document Understanding (Notebooks 05-08)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 05 | `05_image_prompting_and_visual_reasoning.ipynb` | Visual prompting patterns, chart and screenshot analysis, and prompt contracts for image tasks |
| 06 | `06_ocr_layout_and_table_extraction.ipynb` | OCR, layout parsing, table extraction, and why document intelligence is more than just text recognition |
| 07 | `07_structured_outputs_grounding_and_localization.ipynb` | Bounding boxes, grounded answers, schema design, and audit-friendly structured outputs |
| 08 | `08_small_vlms_and_multi_image_workflows.ipynb` | When small VLMs are enough, how to handle multi-image inputs, and how to route across model sizes |

### Module 3: Multimodal RAG and Grounded Retrieval (Notebooks 09-12)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 09 | `09_captioning_as_a_text_bridge.ipynb` | Caption-first multimodal RAG as the simplest bridge from images to text retrieval |
| 10 | `10_page_as_image_retrieval_with_colpali.ipynb` | Page-as-image retrieval, late interaction, and why ColPali matters for document search |
| 11 | `11_hybrid_text_plus_vision_retrieval.ipynb` | Hybrid retrieval, score fusion, reranking, and evidence assembly across text and visual signals |
| 12 | `12_multimodal_grounding_and_evaluation.ipynb` | Groundedness, evidence coverage, localization quality, and multimodal evaluation design |

### Module 4: Audio and Speech Workflows (Notebooks 13-16)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 13 | `13_speech_recognition_and_transcription.ipynb` | Speech recognition, segmentation, timestamp thinking, and Whisper-style transcription pipelines |
| 14 | `14_audio_understanding_classification_and_tagging.ipynb` | Audio event recognition, zero-shot labeling, audio-text embeddings, and CLAP-style audio understanding |
| 15 | `15_speech_plus_document_workflows.ipynb` | How to combine transcripts with pages, slides, and notes for grounded multimodal workflows |
| 16 | `16_multimodal_agents_across_speech_and_vision.ipynb` | Agent loops that can inspect images, consume transcripts, and decide when to call modality-specific tools |

### Module 5: Video and Multimodal Operations (Notebooks 17-19)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 17 | `17_video_understanding_frame_sampling_and_temporal_reasoning.ipynb` | Sampled-frame video understanding, temporal reasoning, and how to stay honest about compute limits |
| 18 | `18_video_grounding_summarization_and_event_extraction.ipynb` | Event extraction, clip grounding, timeline summaries, and short-video analysis patterns |
| 19 | `19_serving_cost_safety_and_evaluation_for_multimodal.ipynb` | Batching, storage, observability, cost accounting, safety review, and multimodal regression discipline |

### Module 6: Capstone - Castalia Perception (Notebooks 20-22)

| # | Notebook | What You'll Build |
|---|----------|-------------------|
| 20 | `20_project_architecture_and_dataset.ipynb` | The architecture, task set, and dataset design for a multimodal Castalia assistant |
| 21 | `21_project_pipeline_and_benchmark.ipynb` | An end-to-end pipeline that combines page understanding, speech transcription, and short-video analysis |
| 22 | `22_project_handoff_and_evaluation_report.ipynb` | A handoff package with grounded outputs, benchmark results, safety notes, and deployment guidance |

---

## Design Philosophy

This course follows four core principles:

1. **Perception Before Polish**: A multimodal system is only useful if it can reliably identify the right evidence before it starts speaking fluently about it.

2. **Open Models Over Black Boxes**: The goal is to understand what image, audio, and video pipelines are doing instead of outsourcing understanding to proprietary endpoints.

3. **Bounded Realism**: We teach methods that are honest about Colab-class hardware. That means sampled frames, manageable clip lengths, smaller model variants, and explicit trade-offs.

4. **Grounding With Consequences**: Multimodal outputs should stay auditable. If a model answers about a chart, page, sound, or clip, we should know what evidence it used and how to evaluate it.

---

## Dependency Philosophy

- **General multimodal reasoning**: `transformers`, `torch`, `accelerate`, `Pillow`
- **Document intelligence and retrieval**: `faiss-cpu`, `sentence-transformers`, `colpali-engine`
- **Audio**: `librosa`, `soundfile`, `torchaudio`, `Whisper`, `CLAP`
- **Video**: `decord`, `opencv-python`, notebook-native frame sampling utilities
- **Measurement and reporting**: `numpy`, `pandas`, `matplotlib`
- **Standard library first**: `json`, `math`, `statistics`, `pathlib`, `dataclasses`, `typing`, `collections`
- **Explicitly avoid hidden orchestration layers**: no framework-first abstractions, no closed multimodal APIs as the teaching path

The goal is to keep every notebook understandable from end to end. Students should see how the media is loaded, how evidence is represented, how prompts are formed, how scores are computed, and why a retrieval or grounding choice helped or hurt.

---

## Recommended Runtime Stack

The course intentionally centers on a small, reusable stack:

- **Qwen2.5-VL** for general image, document, and video reasoning
- **SmolVLM** for smaller and cheaper image-plus-text workflows
- **ColPali** for page-as-image retrieval
- **Whisper** for transcription
- **Qwen2-Audio / CLAP** for richer audio understanding when raw transcription is not enough

This reflects the current open-source state of practice:
- Qwen2.5-VL is one of the most useful open multimodal defaults across images, documents, and short-video tasks
- SmolVLM makes low-resource multimodal experimentation more realistic
- ColPali materially improves document retrieval when layout and page visuals matter
- Whisper remains the practical transcription baseline
- audio-text models and retrieval embeddings become important once the task is about sounds, not just spoken words

---

## Castalia Perception Capstone

The overarching project for this track is **Castalia Perception**: a multimodal assistant that can inspect pages, answer grounded questions about images, transcribe and analyze short audio clips, and summarize short videos with explicit evidence and benchmark outputs.

Across the capstone sequence, students will:
- define a multimodal task set
- assemble a small benchmark dataset
- wire together page understanding, transcript processing, and video summaries
- compare alternative retrieval and grounding choices
- produce an evaluation and deployment handoff

The point of the capstone is not merely to call a multimodal model. It is to build a measurable multimodal workflow that connects directly to the rest of Castalia.

---

## How to Use This Course

### Recommended Path After `systems/`
1. Finish [prompt-engineering/](../prompt-engineering/) to learn control and decomposition
2. Finish [rag/](../rag/) to learn retrieval, grounding, and evidence-aware generation
3. Finish [agents/](../agents/) to learn orchestration and tool use
4. Finish [evals/](../evals/) to learn measurement and regression discipline
5. Finish [finetuning/](../finetuning/) to learn adaptation and post-training
6. Finish [systems/](../systems/) to learn runtime engineering and deployment discipline
7. Work through `multimodal/` in order: Module 1 -> Module 2 -> Module 3 -> Module 4 -> Module 5 -> Module 6

### What This Track Adds
- It extends Castalia from text-centric systems into perception-centric systems
- It turns image, page, audio, and video evidence into auditable workflows
- It connects multimodal prompting, retrieval, evaluation, and deployment instead of treating them as separate demos
- It gives a realistic open-source path rather than a closed API showcase

### What Comes After `multimodal/`
After this track, learners should be prepared for:
- deeper multimodal finetuning and preference optimization
- richer multimodal agent environments
- stronger multimodal serving and observability systems
- continual improvement loops that span text, image, audio, and video

---

## Course Status

The `multimodal/` track is now created as a 22-notebook course focused on open-source multimodal reasoning, document intelligence, audio understanding, short-video workflows, and benchmark-driven deployment discipline. It is intended to be worked through after the existing Castalia stack and acts as the perception layer for the broader curriculum.
