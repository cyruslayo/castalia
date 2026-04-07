# Systems — Open-Source Inference, Serving, and Runtime Engineering

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
