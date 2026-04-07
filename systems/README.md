# Systems — Open-Source Inference, Serving, and Runtime Engineering

**A notebook-based course on modern open-model inference systems, serving runtimes, routing, observability, and deployment engineering for 2026.**

This folder will contain 22 Jupyter notebooks organized into 6 modules, progressing from runtime fundamentals to a full serving and benchmarking capstone. The focus is not general distributed systems theory. It is the practical systems layer that comes **after finetuning**: how to run open models efficiently, measure them honestly, expose them safely, route requests intelligently, and ship them with observability and regression discipline.

| Component | Implementation |
|---|---|
| **Local Runtime** | llama.cpp / GGUF / notebook-native baselines |
| **Serving Runtimes** | vLLM + SGLang |
| **API Layer** | FastAPI + OpenAI-compatible patterns |
| **Benchmarking** | Pure Python timing, throughput, and quality harnesses |
| **Artifacts** | JSON/CSV benchmark reports, routing manifests, deployment handoff files |
| **Framework** | Notebook-first, open-source only, minimal abstractions |

> **Prerequisites:** Complete [prompt-engineering/](../prompt-engineering/), [rag/](../rag/), [agents/](../agents/), [evals/](../evals/), and [finetuning/](../finetuning/) first. This track assumes you already know how to control, ground, orchestrate, evaluate, and post-train models.
> **Runtime:** Colab-friendly where practical, but this course is intentionally more systems-oriented than earlier tracks. Some notebooks focus on concepts and local benchmarks, while others show runnable serving patterns that are easiest on Linux or GPU-backed environments.

---

## Why This Track Exists

The earlier Castalia tracks teach how to:
- control model behavior with prompts
- ground outputs with retrieval
- orchestrate tools and agents
- measure quality and regressions
- change model behavior through finetuning

This track teaches what comes next: **how to run those models as systems**.

In other words:
- `prompt-engineering/` teaches **control**
- `rag/` teaches **grounding**
- `agents/` teaches **orchestration**
- `evals/` teaches **measurement**
- `finetuning/` teaches **behavior change**
- `systems/` teaches **runtime engineering and deployment discipline**

That shift matters. A strong model is not automatically a strong product system. Once a model is selected or finetuned, the hard questions become:
- how requests are batched
- how memory is managed
- how latency and throughput trade off
- how structured outputs are enforced
- how adapters are served
- how failures, retries, and overload conditions are handled
- how regressions are caught before deployment

---

## Course Structure

### Module 1: Foundations and Measurement (Notebooks 01–04)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_intro_to_llm_systems_2026.ipynb` | What changes between models and systems, where systems work begins after finetuning, and the quality/latency/cost/reliability tradeoff |
| 02 | `02_runtime_landscape_and_deployment_tiers.ipynb` | Transformers baselines, llama.cpp, vLLM, SGLang, and the mental model for local vs single-node vs cluster serving |
| 03 | `03_latency_throughput_and_memory_budgeting.ipynb` | Request lifecycle, prefill vs decode, batching effects, token-rate thinking, and KV cache budgeting |
| 04 | `04_building_a_reproducible_benchmark_harness.ipynb` | Timing harnesses, throughput harnesses, memory tracking, repeatable benchmark datasets, and report generation |

### Module 2: Local and Single-Node Systems (Notebooks 05–08)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 05 | `05_gguf_and_llama_cpp_basics.ipynb` | GGUF, llama.cpp, local inference, CPU/GPU offload, and practical local runtime setup |
| 06 | `06_quantization_and_memory_tradeoffs.ipynb` | Quantization formats, memory-speed-quality tradeoffs, and how runtime constraints shape model choice |
| 07 | `07_local_openai_compatible_serving.ipynb` | Local API serving, request/response contracts, and OpenAI-compatible runtime interfaces |
| 08 | `08_transformers_baselines_and_batching.ipynb` | Raw Transformers serving baselines, manual batching, async request handling, and why optimized runtimes exist |

### Module 3: Modern Serving Runtimes (Notebooks 09–13)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 09 | `09_vllm_quickstart_and_api_serving.ipynb` | vLLM setup, OpenAI-compatible serving, throughput mindset, and first runtime benchmarks |
| 10 | `10_continuous_batching_and_paged_kv_cache.ipynb` | Continuous batching, managed KV cache, request interleaving, and runtime-level throughput gains |
| 11 | `11_prefix_caching_chunked_prefill_and_long_context.ipynb` | Prefix reuse, chunked prefill, long-context serving patterns, and cache-aware prompt design |
| 12 | `12_structured_outputs_tool_calls_and_response_contracts.ipynb` | JSON/schema-constrained generation, tool-call response shapes, parser layers, and contract safety |
| 13 | `13_multi_lora_serving_and_runtime_model_selection.ipynb` | Serving adapters, multi-LoRA patterns, model routing, and runtime selection heuristics |

### Module 4: Advanced Runtime Patterns (Notebooks 14–17)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 14 | `14_sglang_and_advanced_scheduling_patterns.ipynb` | SGLang, advanced scheduling concepts, throughput-oriented serving, and runtime comparisons |
| 15 | `15_speculative_decoding_and_assisted_generation.ipynb` | Speculative decoding, draft models, acceptance rates, and when the extra complexity is worth it |
| 16 | `16_routing_retries_admission_control_and_load_shedding.ipynb` | Overload behavior, backpressure, retries, admission control, and graceful degradation |
| 17 | `17_distributed_inference_concepts.ipynb` | Tensor, pipeline, data, and expert parallelism as practical runtime ideas rather than abstract theory |

### Module 5: Reliability, Evals, and Operations (Notebooks 18–19)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 18 | `18_observability_metrics_tracing_and_cost_accounting.ipynb` | Metrics, traces, logs, request accounting, resource dashboards, and cost-aware benchmarking |
| 19 | `19_eval_driven_rollouts_regressions_and_incident_playbooks.ipynb` | Canary rollouts, benchmark-gated releases, runtime regressions, rollback plans, and incident debugging |

### Module 6: Capstone — Castalia Runtime (Notebooks 20–22)

| # | Notebook | What You'll Build |
|---|----------|-------------------|
| 20 | `20_project_runtime_prototype.ipynb` | A notebook-native serving prototype for Castalia Mentor with local and optimized runtime paths |
| 21 | `21_project_benchmark_routing_and_structured_outputs.ipynb` | Runtime comparison, routing logic, structured outputs, and benchmark-driven decisions |
| 22 | `22_project_deployment_handoff.ipynb` | Deployment handoff, observability checklist, regression gates, and export-ready system manifests |

---

## Design Philosophy

This course follows four core principles:

1. **Runtime Behavior First**: We do not treat serving as a black box. We expose the mechanics that drive latency, throughput, memory use, and reliability.

2. **Open-Source Runtimes Over Managed Platforms**: Students should understand how the serving stack works without depending on proprietary infrastructure or hidden vendor logic.

3. **Benchmarks With Consequences**: System optimizations only matter if quality and contract correctness survive them. Every serious runtime change should be benchmarked.

4. **Deployment Thinking Without Platform Worship**: The target is not to memorize one company stack. The target is to understand the patterns that transfer across runtimes and environments.

The underlying lesson is simple: model quality is necessary, but systems quality determines whether the model is actually usable.

---

## Dependency Philosophy

- **Primary runtime references**: `llama.cpp`, `llama-cpp-python`, `vllm`, and `SGLang`
- **API and contract layer**: `fastapi`, `uvicorn`, `pydantic`, `httpx`
- **Measurement stack**: `numpy`, `pandas`, `matplotlib`
- **Transformers baseline**: `transformers`, `torch`, `accelerate`
- **Standard library first**: `json`, `time`, `math`, `statistics`, `asyncio`, `threading`, `collections`, `pathlib`, `dataclasses`, `typing`
- **Explicitly avoid opaque managed serving platforms**: no proprietary inference dashboards, no vendor routing layers as the main teaching path

The goal is to keep every notebook understandable from end to end. Students should know where the queue lives, where the cache lives, what the batcher is doing, and why a serving choice helped or hurt.

---

## Recommended Runtime Stack

The course is intentionally centered on a small number of runtimes:

- **llama.cpp / llama-cpp-python** for local and edge-friendly inference
- **vLLM** for high-throughput serving and modern runtime mechanics
- **SGLang** for advanced serving patterns and comparative runtime reasoning

This choice reflects the current state of practice:
- **vLLM** is a central open inference runtime for throughput-oriented serving
- **SGLang** is a major serving and post-training runtime with strong scheduling and caching features
- **TGI** is now useful as ecosystem context, but not the main path
- **llama.cpp** remains essential for low-dependency, local, and GGUF-based serving

---

## Castalia Runtime Capstone

The overarching project for this track is **Castalia Runtime**: a notebook-first open-model serving and benchmarking stack for the broader Castalia curriculum.

Across the capstone sequence, students will:
- define a serving contract for `Castalia Mentor`
- stand up a local runtime path
- add an optimized serving path
- benchmark multiple runtime strategies
- introduce structured outputs and routing
- prepare a deployment and observability handoff

The point of the capstone is not merely to run a model. It is to build a serving layer that is measurable, inspectable, and ready for later multimodal extension.

---

## How to Use This Course

### Recommended Path After `finetuning/`
1. Finish [prompt-engineering/](../prompt-engineering/) to learn control and decomposition
2. Finish [rag/](../rag/) to learn grounding and evidence-aware generation
3. Finish [agents/](../agents/) to learn orchestration and tool use
4. Finish [evals/](../evals/) to learn measurement and regression discipline
5. Finish [finetuning/](../finetuning/) to learn post-training and alignment
6. Work through `systems/` in order: Module 1 → Module 2 → Module 3 → Module 4 → Module 5 → Module 6

### What This Track Adds
- It turns post-trained models into deployable systems
- It explains why runtime design changes model behavior in practice
- It teaches realistic 2026 open-source serving patterns instead of idealized demos
- It builds the operational bridge from model engineering into multimodal systems

### What Comes After `systems/`
After this track, learners should be prepared for:
- the dedicated [multimodal/](../multimodal/) track
- multimodal runtime design
- multimodal serving and evaluation
- richer tool and perception loops
- broader deployment, monitoring, and continual improvement systems

---

## Course Status

The `systems/` track is now fully created as a 22-notebook course focused on open-source inference, serving, routing, observability, and deployment systems. The folder now contains the complete sequence from runtime foundations through modern serving patterns, operational reliability, and the three-part Castalia Runtime capstone.
