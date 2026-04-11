# Foundations — First Principles Before Frameworks

> **Orbit -1** · 7 notebooks · Prerequisites: None

This module builds the conceptual bedrock for the entire Castalia curriculum. Instead of teaching tools, it derives the *why* behind every technique. Complete this orbit to go from "I know how to use it" to "I understand why it works."

## Course Structure

| # | Notebook | Topics | Difficulty |
|---|----------|--------|------------|
| 00 | [00_how_llms_work.ipynb](00_how_llms_work.ipynb) | Transformer mental models, next-token prediction, generation pipeline | Beginner |
| 01 | [01_tokenization_deep_dive.ipynb](01_tokenization_deep_dive.ipynb) | BPE algorithm, vocabulary effects, multilingual tokenization, token economics | Beginner |
| 02 | [02_embeddings_and_vector_spaces.ipynb](02_embeddings_and_vector_spaces.ipynb) | Embedding geometry, cosine similarity, anisotropy, MTEB intuition | Intermediate |
| 03 | [03_sampling_and_decoding.ipynb](03_sampling_and_decoding.ipynb) | Temperature, top-k, top-p, beam search, repetition penalties | Intermediate |
| 04 | [04_information_retrieval_theory.ipynb](04_information_retrieval_theory.ipynb) | TF-IDF, BM25, inverted indices, lexical vs semantic retrieval | Intermediate |
| 05 | [05_attention_and_context.ipynb](05_attention_and_context.ipynb) | Self-attention from scratch, multi-head attention, KV cache, context windows | Intermediate |
| 06 | [06_scaling_laws_and_model_selection.ipynb](06_scaling_laws_and_model_selection.ipynb) | Scaling laws, VRAM budgeting, quantization, model selection framework | Intermediate |

## How to Use This Module

**If you're new to AI engineering**: Start with notebook 00 and work through sequentially. Each notebook builds on the previous.

**If you have some experience**: Use these as reference. When a notebook in Orbits 0–7 introduces a concept you want to understand more deeply, come back to the relevant foundations notebook.

**Key principle**: Every algorithm is derived before a library is used. You'll implement TF-IDF before calling scikit-learn, build attention from scratch before using `transformers`, and calculate VRAM budgets by hand before relying on tools.

## Environment

All notebooks run on **Google Colab Pro** with a T4 GPU. Most foundations notebooks are CPU-only for the core content, with optional GPU sections for model demonstrations.
