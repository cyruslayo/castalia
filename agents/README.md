# Agentic AI — A Master Course in Agent Design & Engineering

**A complete, first-principles course on building AI agents from scratch using open-source models.**

This folder contains 37 Jupyter notebooks organized into 7 modules, progressing from foundational concepts to a complete multi-agent research system. Every pattern is built from scratch — no LangChain, no LangGraph, no CrewAI, no frameworks. You will understand *exactly* how each component works.

| Component | Implementation |
|---|---|
| **LLM** | Qwen/Qwen3-14B (4-bit NF4 quantization) |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dim, sentence-transformers) |
| **Vector Store** | FAISS (faiss-cpu) with numpy arrays |
| **Graph Store** | networkx |
| **Framework** | Pure Python — zero abstraction layers |

> **Prerequisites:** Complete the [prompt-engineering/](../prompt-engineering/) and [rag/](../rag/) modules first. This course builds directly on those foundations.
> **Runtime:** Google Colab with T4 GPU (free tier works). Each notebook is self-contained and independently runnable.

---

## Course Structure

### Module 1: Foundations of Agentic AI (Notebooks 01–05)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | `01_intro_to_agentic_ai.ipynb` | What agents are, the agentic spectrum (L0–L4), history from expert systems to LLM agents, perception-action loops |
| 02 | `02_the_agent_loop.ipynb` | The universal agent loop, `AgentState` dataclass, `AgentLoop` class, termination strategies |
| 03 | `03_tool_use_and_function_calling.ipynb` | Tool schemas, `ToolRegistry` with auto-schema extraction, tool dispatching, error handling |
| 04 | `04_structured_output_parsing.ipynb` | Reliable JSON/XML parsing from LLMs, schema validation, retry-with-feedback, output repair |
| 05 | `05_building_a_react_agent.ipynb` | The ReAct pattern (Thought→Action→Observation), complete from-scratch implementation |

### Module 2: Single Agent Patterns (Notebooks 06–12)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 06 | `06_plan_and_execute.ipynb` | Separating planning from execution, adaptive re-planning, plan-as-DAG |
| 07 | `07_reflection_and_self_critique.ipynb` | Generate→Critique→Revise loops, configurable rubrics, quality tracking |
| 08 | `08_tree_of_thought.ipynb` | Multiple reasoning paths, BFS/DFS exploration, path evaluation, pruning |
| 09 | `09_iterative_refinement.ipynb` | Draft→Feedback→Revise cycles, convergence detection, cost-benefit analysis |
| 10 | `10_agent_memory_short_term.ipynb` | Conversation memory, sliding windows, summarization, importance-weighted retention |
| 11 | `11_agent_memory_long_term.ipynb` | Vector-based persistent memory with FAISS + BGE, episodic & semantic storage |
| 12 | `12_knowledge_graph_memory.ipynb` | Graph-based memory with networkx, entity-relation extraction, multi-hop queries |

### Module 3: Tool Engineering (Notebooks 13–16)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 13 | `13_advanced_tool_design.ipynb` | Production-quality tools, validation, error recovery, tool testing harness |
| 14 | `14_code_execution_tool.ipynb` | Safe code generation & execution, sandboxing, timeout, import whitelisting |
| 15 | `15_web_and_search_tools.ipynb` | Information retrieval tools, content extraction, result ranking |
| 16 | `16_file_and_data_tools.ipynb` | CSV/JSON processing, data analysis via tools, statistics tools |

### Module 4: Multi-Agent Systems (Notebooks 17–23)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 17 | `17_multi_agent_conversation.ipynb` | Two agents communicating, message passing, turn-taking, shared context |
| 18 | `18_agent_debate_and_consensus.ipynb` | Multi-agent debate, structured argumentation, judge agents, voting |
| 19 | `19_sequential_agent_pipelines.ipynb` | Chains of specialized agents, interface contracts, error propagation |
| 20 | `20_hierarchical_agent_delegation.ipynb` | Manager-worker patterns, task routing, result aggregation |
| 21 | `21_agent_orchestration_patterns.ipynb` | Router agents, conditional routing, DAG-based execution, graph executor |
| 22 | `22_shared_state_and_blackboard.ipynb` | Blackboard architecture, shared state, event-driven agents |
| 23 | `23_swarm_intelligence.ipynb` | Emergent behavior from simple agents, population-based exploration |

### Module 5: Production Concerns (Notebooks 24–28)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 24 | `24_agent_safety_and_guardrails.ipynb` | Prompt injection defense, tool validation, PII detection, rate limiting |
| 25 | `25_human_in_the_loop.ipynb` | Approval gates, feedback loops, escalation, interactive checkpoints |
| 26 | `26_agent_evaluation_and_testing.ipynb` | Metrics, golden datasets, LLM-as-judge, regression testing |
| 27 | `27_cost_and_latency_optimization.ipynb` | Token budgeting, caching, model routing, early termination |
| 28 | `28_error_handling_and_resilience.ipynb` | Retry strategies, fallback chains, circuit breakers, graceful degradation |

### Module 6: Communication Protocols (Notebooks 29–31)

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 29 | `29_mcp_from_scratch.ipynb` | Model Context Protocol — build a server & client from scratch |
| 30 | `30_a2a_protocol.ipynb` | Agent-to-Agent communication, agent cards, capability discovery |
| 31 | `31_building_an_agent_runtime.ipynb` | Complete agent runtime with lifecycle management and observability |

### Module 7: Capstone Project — Castalia Scholar (Notebooks 32–37)

Build a complete **multi-agent research assistant** that uses every pattern from the course.

| # | Notebook | What You'll Build |
|---|----------|-------------------|
| 32 | `32_project_architecture.ipynb` | System architecture, component interfaces, data flow design |
| 33 | `33_project_retrieval_agent.ipynb` | Information retrieval agent with Agentic RAG, citation tracking |
| 34 | `34_project_analysis_agent.ipynb` | Analysis agent with source synthesis, contradiction detection |
| 35 | `35_project_writing_agent.ipynb` | Report generation agent with section planning, citation integration |
| 36 | `36_project_review_agent.ipynb` | Quality review agent with multi-dimensional evaluation |
| 37 | `37_project_full_system.ipynb` | Full orchestration — all agents working together end-to-end |

---

## How to Use This Course

### Recommended Path
1. Complete [prompt-engineering/](../prompt-engineering/) (especially: intro, CoT, task decomposition, prompt chaining)
2. Complete [rag/](../rag/) (especially: simple RAG, agentic RAG, self-RAG, CRAG, graph RAG)
3. Work through this folder in order: Module 1 → 2 → 3 → 4 → 5 → 6 → 7

### Each Notebook is Self-Contained
- Every notebook can be opened and run independently in Google Colab
- Setup cells install all dependencies and load the model
- No notebook requires files generated by previous notebooks (though concepts build sequentially)

### What You Need
- A Google account (for Colab)
- Free Colab T4 GPU runtime
- ~8GB free Google Drive space (for model weight caching)
- No API keys, no paid services, no external accounts

---

## Key Research References

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| ReAct: Synergizing Reasoning and Acting | Yao et al. | 2022 | Foundation of the ReAct agent pattern |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al. | 2023 | Self-critique and reflection |
| Tree of Thoughts: Deliberate Problem Solving | Yao et al. | 2023 | Multi-path reasoning |
| Toolformer: Language Models Can Teach Themselves to Use Tools | Schick et al. | 2023 | Tool use foundations |
| Generative Agents: Interactive Simulacra of Human Behavior | Park et al. | 2023 | Memory systems for agents |
| Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | Iterative improvement |
| Improving Factuality through Multiagent Debate | Du et al. | 2023 | Multi-agent reasoning |
| A Survey on LLM-based Autonomous Agents | Wang et al. | 2024 | Comprehensive survey |
| Judging LLM-as-a-Judge | Zheng et al. | 2023 | Agent evaluation |

---

## Design Philosophy

This course follows three core principles:

1. **First Principles Over Frameworks**: Every agent pattern is built from raw Python. You understand the internals before ever touching a framework.

2. **Open Source Only**: Qwen3-14B runs locally on a free GPU. No API keys, no vendor lock-in, no usage costs. Fully reproducible.

3. **Understanding Over Usage**: We don't just show *how* to build agents — we explain *why* each design works, *when* it fails, and *what* the trade-offs are.
