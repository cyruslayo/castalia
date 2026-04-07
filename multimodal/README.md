# Multimodal - Vision, Document Intelligence, Audio, and Video Systems

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
