# Castalia Agent Course — Complete Progress Log

> **Created:** 2026-05-06
> **Purpose:** Reference document for the domain-tutor to continue lessons from where we left off. Maps the full 37-notebook curriculum against what has been built in the custom `build-my-agent/` project.
> **Current Level:** 0 (Complete beginner, heavy scaffolding required)
> **API:** OpenAI-compatible vLLM endpoint (hermes-model)
> **Working Directory:** `C:/AI2026/castalia/build-my-agent/`

---

## Part 1: Full Curriculum Map (37 Notebooks, 7 Modules)

### Module 1: Foundations of Agentic AI (Notebooks 01–05)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| 01 | `intro_to_agentic_ai` | What makes a system "agentic", 5-level autonomy spectrum (L0-L4), perception-action loop, formal definition (Memory + Goals + Tools + Autonomy), history from expert systems to LLM agents, "an agent is a loop not a model" | ✅ Covered |
| 02 | `the_agent_loop` | Universal perceive→reason→act cycle, `AgentState` dataclass, `AgentLoop` class with `run()` and `step()` methods, JSON parsing with fallback, 4 termination strategies (max_steps, self_terminate, convergence, combined), agent as finite state machine | ✅ Built: state.py, agent_loop.py, parser.py |
| 03 | `tool_use_and_function_calling` | Tool schemas, ParameterSchema + Tool dataclasses, ToolRegistry with auto-schema extraction, ToolDispatcher (validate + execute), 8 tools, error handling, build_tool_system_prompt | ✅ Built: tools.py with 5 tools, structured schemas, security gates, subprocess isolation |
| 04 | `structured_output_parsing` | LLMs produce malformed JSON 10-30% of the time, 4 extraction strategies (regex, JSON repair, XML tags, delimiters), OutputParser unified, schema validation (Field + Schema), RetryParser (feeds errors back to LLM), full parse→validate→clean pipeline | ✅ Built: parser.py (4 strategies) + guard.py Layer 2 (schema validation, type checking) + Layer 3 (semantic recovery, 75+ tests) |
| 05 | `building_a_react_agent` | ReAct paradigm (Thought→Action→Observation interleaving), 8 tools with 30-fact knowledge base, system prompt design, single prompt vs CoT vs ReAct comparison, failure analysis, optimization tips (temperature, prompt length, token budgeting) | ✅ Built: react_agent.py (full ReAct agent with cycles), knowledge_base.py (23 facts, keyword search), comparison_test.py |

### Module 2: Single Agent Patterns (Notebooks 06–12)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| 06 | `plan_and_execute` | Why plan first (reactive agents fail on dependencies), Planner vs Executor separation, PlanStep (id, description, dependencies, status, result), Plan (get_ready_steps, dependency resolution), PlannerAgent (create_plan, replan), ExecutorAgent, PlanAndExecuteAgent coordinator, adaptive re-planning, when to use (complex >3-4 steps, expensive errors, order matters) | ✅ Built: plan_agent.py (PlanStep, Plan, Planner, PlanAndExecuteAgent, dependency resolution, synthesis) |
| 07 | `reflection_and_self_critique` | Generate→Critique→Revise loop (Madaan et al. 2023), configurable rubrics (RubricDimension with name/weight/criteria, Rubric), CritiqueResult (parse, weighted score, summary), ReflectionAgent (generate/critique/revise/run, score_trajectory), convergence detection, when reflection hurts (simple tasks, over-editing, critique hallucination) | ✅ Built: reflection_agent.py (Critique dataclass, critic prompt, _extract_json + _parse_natural_response fallback, revision prompt, revise_answer temp=0.7, ReflectionAgent composable wrapper over any inner agent) |
| 08 | `tree_of_thought` | Why linear reasoning fails (wrong first step, multiple approaches, backtracking, exploration), 3 paradigms (CoT→Self-Consistency→ToT), Yao et al. 2023, ThoughtNode (content, score, parent, children, get_path, path_text), TreeOfThought engine (generate, evaluate, BFS, DFS, two-stage pruning: threshold + top_k, early termination ≥9.0), cost analysis | ✅ Built: tree_of_thought.py (ThoughtNode, generate_thoughts temp=0.8, evaluate_thought temp=0.3, BFS/DFS, two-stage pruning, early termination, reasoning field fallback, max_tokens=1024, BFS: 8 calls/17 nodes, DFS: 4 calls/4 nodes) |
| 09 | `iterative_refinement` | **Replace subjective self-critique with external pluggable feedback.** Feedback the LLM cannot fabricate: unit tests, style scorers, fact checkers. FeedbackResult (score, passed, feedback_text, details), RefinementIteration, RefinementTrace (score_trajectory, summary, total_time). IterativeRefinementAgent: generate_initial, revise, check_convergence (3 detectors: score threshold, plateau, decline). 3 use cases: (1) Code with test feedback (safe_exec, dynamic harness), (2) Text style (deterministic heuristics), (3) Fact checking (claim extraction, DB lookup) | ✅ Built: iterative_refinement.py (FeedbackResult, RefinementIteration, RefinementTrace, IterativeRefinementAgent, safe_exec, code_test_feedback, style_feedback, fact_check_feedback, 3-stage convergence, 5 fast tests pass, 1 LLM test 10.0/10) |
| **10** | **`agent_memory_short_term`** | **The memory problem. Strategy 1: Full History (perfect, unbounded). Strategy 2: Sliding Window (bounded, hard cutoff loss). Strategy 3: Summarization (old compressed, costs tokens). Strategy 4: Importance-Weighted (heuristic scoring). MemoryManager unified controller. 30-turn stress test.** | ✅ Built: memory.py (4 strategies, MemoryManager, stress test) + agent_loop.py integration |
| **11** | **`agent_memory_long_term`** | **Beyond context window. Episodic memory (timestamped experiences with embeddings) vs semantic memory (deduplicated fact store). LongTermMemory unified (store_episode, store_fact, recall, consolidate). FAISS + BGE. Cross-session persistence. Importance decay.** | ✅ Built: long_term_memory.py (5 sections, BGE 384-dim embeddings, FAISS IndexFlatIP, decay, consolidation, JSON persistence) |
| **12** | **`knowledge_graph_memory`** | **Why graphs complement vectors. Multi-hop, relationship, structural queries. Entity-relation extraction (triples). GraphMemory (add_memory, add_triple, query, multi_hop_query, visualize). networkx. Hybrid (vectors + graph). Maintenance (contradictions, redundancy, staleness).** | ✅ Built: graph_memory.py (GraphMemory with nx.DiGraph, entity canonicalization, BFS multi-hop; LLMQueryClassifier Option B with caching; HybridMemory with vector+graph dual-write, LLM-routed queries; GraphMaintenance with contradiction detection + health reports) |

### Module 3: Tool Engineering (Notebooks 13–16)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| **13** | **`advanced_tool_design`** | **Production tools vs naive (no validation, no error handling, no docs, no testing). Design principles: single responsibility, clear schemas, predictable errors. ToolDefinition dataclass. Stateful, confirmation, composed tool types. Input validation with LLM-friendly errors. Testing harness.** | ✅ Built: tool_registry.py, production_tools.py, agent_tool_integration.py, test_production_llm.py |
| **14** | **`code_execution_tool`** | **Production code execution with 5-layer security pipeline. AST-based static analysis, subprocess sandbox with hard kill, Docker-ready architecture, output sanitization (PII redaction), audit logging, rate limiting. CodeExecutor (full pipeline), CodeAgent (ReAct self-correction), comparison runner.** | ✅ Built: static_analyzer.py, output_sanitizer.py, audit_logger.py, code_sandbox.py (SubprocessSandbox + DockerSandbox), code_executor.py (5-layer pipeline), code_agent.py (write-run-fix loop, retry, comparison) |
| **15** | **`web_and_search_tools`** | **Why search (post-cutoff, private, verify, cite). Document corpus (30 articles). TF-IDF from scratch. FAISS semantic search. TF-IDF vs semantic head-to-head. Content extraction. Research agent.** | ✅ Built: web_search_tools.py (Tavily API client with dataclasses, error handling, registry integration), tavily_research_agent.py (direct search→synthesize pipeline with citations), demo_tavily_agent.py (ReAct with web_search, tuned prompt with CRITICAL RULES forcing synthesis after 1-2 searches), multi_query_tavily_agent.py (LLM-driven query reformulation + URL-deduplicated merge), notebook15_corpus.py (30-doc local corpus from Notebook 15), local_search.py (pure-Python TF-IDF engine from scratch: tokenization, indexing, min-max normalization), semantic_local_search.py (BGE via fastembed ONNX + FAISS IndexFlatIP, auto-fallback to TF-IDF cosine if BGE unavailable), hybrid_search.py (3-source fusion: Tavily web 0.4 + local TF-IDF 0.2 + local semantic 0.4, min-max normalized, merged ranking) |
| **16** | **`file_and_data_tools`** | **Data-centric tasks (read, query, transform, compute, write). VirtualFS in-memory filesystem. CSV/JSON/Statistics tools from scratch. Data analysis agent on countries dataset.** | ✅ Built: schema_aware_fs.py (6-layer metadata, schema inference, derivation chains), unified_data_tools.py (consolidated DataReader/DataQuery/DataStats/DataTransform, auto-type detection, SQL-like unified query), self_correcting_data_agent.py (anomaly detection, self-correction, memory integration, workflow templates), demo_data_agent.py (countries dataset, 5/5 evaluation cases, LLM synthesis with timeout) |

### Module 4: Multi-Agent Systems (Notebooks 17–23)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| **17** | **`multi_agent_conversation`** | **Specialization, verification, natural roles, debate. Message (sender, recipient, content, timestamp, metadata). ConversableAgent (personality, own history, response generation). TwoAgentChat (turn-taking, termination). Experiments: Researcher+Skeptic, Teacher+Student.** | ✅ Built: multi_agent_conversation.py |
| **18** | **`agent_debate_and_consensus`** | **Debate pattern (Du et al. 2023). Diverse initial answers, structured arguments, DebaterAgent, JudgeAgent, DebateArena (multi-round, voting, final decision).** | ✅ Built: multi_agent_debate.py |
| 19 | `sequential_agent_pipelines` | **Unix pipes analogy. Specialized stages outperform generalists. Message dataclass, AgentNode, Pipeline (chain N agents, logging, error, partial execution). Research pipeline example.** | ✅ Built: sequential_pipelines.py, sequential_walkthrough.py |
| 20 | `hierarchical_agent_delegation` | **Manager-worker (Park et al. 2023). Decompose-Delegate-Aggregate. WorkerAgent, WorkerRegistry. Manager decomposes, delegates, aggregates. Flat vs hierarchical.** | ✅ Built: hierarchical_delegation.py |

| **21** | **`agent_orchestration_patterns`** | **Routing and control flow. Router Agent, conditional routing, parallel fan-out/fan-in, DAG-based (topological sort).** | ✅ Done |
| **22** | **`shared_state_and_blackboard`** | **Shared workspace, knowledge sources, controller. Advantages over message-passing. Blackboard class, BlackboardAgent, event-driven, conflict resolution.** | ⬜ Future |
| **23** | **`swarm_intelligence`** | **Emergent behavior from simple agents. Decentralization, local interaction, stigmergy, emergence. SimpleSwarmAgent, SwarmCoordinator, diversity analysis.** | ⬜ Future |

### Module 5: Production Concerns (Notebooks 24–28)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| **24** | **`agent_safety_and_guardrails`** | **Prompt injection, PII detection, tool validation (whitelist, bounds, types), rate limiting, audit logging.** | ⬜ Future |
| **25** | **`human_in_the_loop`** | **Approval gates, feedback loops, escalation, interactive mode, human intervenes at any step.** | ⬜ Future |
| **26** | **`agent_evaluation_and_testing`** | **Non-deterministic, multi-step, subjective. Metrics, golden dataset (15+ tasks), automated scoring, LLM-as-judge, regression testing.** | ⬜ Future |
| **27** | **`cost_and_latency_optimization`** | **Token budget, caching, prompt compression, model routing, circularity detection, early termination.** | ⬜ Future |
| **28** | **`error_handling_and_resilience`** | **RetryWithFeedback, FallbackChain, CircuitBreaker (CLOSED→OPEN→HALF_OPEN), GracefulDegradation, timeout.** | ⬜ Future |

### Module 6: Communication Protocols (Notebooks 29–31)

| # | Notebook | Core Concepts | Custom Build Status |
|---|----------|---------------|---------------------|
| **29** | **`mcp_from_scratch`** | **Model Context Protocol (Anthropic 2024). Host/Client/Server roles, JSON-RPC, MCP server, MCP client, handshake, calculator example.** | ⬜ Future |
| **30** | **`a2a_protocol`** | **Agent-to-Agent (Google 2025). Agent Card (identity/capabilities/endpoint/auth), Agent Directory, Task Protocol, 3-agent collaboration.** | ⬜ Future |
| **31** | **`building_an_agent_runtime`** | **Infrastructure: no visibility, no coordination, no recovery, no management. EventBus (pub-sub), AgentLogger (structured, traces, JSON), AgentRegistry, AgentLifecycle (start/stop/restart/health check), complete runtime.** | ⬜ Future |

### Module 7: Capstone — Castalia Scholar (Notebooks 32–37)

| # | Notebook | What You'll Build | Custom Build Status |
|---|----------|-------------------|---------------------|
| **32** | **`project_architecture`** | **System architecture, component interfaces, data flow, agent roles (Retriever, Analyzer, Writer, Reviewer, Orchestrator).** | ⬜ Future |
| **33** | **`project_retrieval_agent`** | **40+ research documents, BGE embeddings, FAISS, retrieval agent with semantic search, relevance filtering, citation tracking.** | ⬜ Future |
| **34** | **`project_analysis_agent`** | **Source synthesis, contradiction detection, evidence weighing, confidence scoring, analysis pipeline.** | ⬜ Future |
| **35** | **`project_writing_agent`** | **Report generation, section planning, evidence integration, citation manager, clear prose, academic tone, structured output.** | ⬜ Future |
| **36** | **`project_review_agent`** | **Multi-dimensional quality rubric, factual verification, prioritized feedback (location/issue/fix/severity), pass/revise decision.** | ⬜ Future |
| **37** | **`project_full_system`** | **Full orchestration: all agents together end-to-end. Planner decomposes, Retriever finds docs, Analyzer synthesizes, Writer drafts, Reviewer checks, Orchestrator coordinates with blackboard.** | ⬜ Future |

---

## Part 2: Custom Build Archive — What Was Actually Built

All files live in `C:/AI2026/castalia/build-my-agent/`

### Core Modules (Notebooks 01-05 equivalent)

#### state.py (2177 bytes)
- `AgentState` @dataclass with 8 fields: goal, messages, steps, current_step, max_steps, is_complete, final_answer, metadata
- Helper methods: `elapsed_time()`, `summary()`
- Critical: `default_factory=list` for messages (avoids shared mutable state)

#### agent_loop.py (12393 bytes)
- Core execution engine: `step()` method with system prompt + memory + LLM call + parser
- Imports `guarded_execute_tool` from guard.py (full 3-layer defense)
- Action dispatch: "think" (just records thought), "answer" (sets final_answer, marks complete), "use_tool" (runs guard pipeline)
- Real LLM integration via config.py (OpenAI client, vLLM endpoint)
- max_tokens=2048, temperature=0.7 (fixed 400 context window error)
- 3-step workflow validated: "Calculate 987*654, write to file, read back" (6 steps total)

#### parser.py (15386 bytes)
- 4-strategy fallback cascade:
  1. `try_direct_parse`: `json.loads(text)`
  2. `try_markdown_extract`: regex for code blocks with JSON
  3. `try_brace_extract`: find matching `{...}` and parse
  4. `try_keyword_extract`: scan for "action", "thought", "tool" keywords
- Failsafe: returns `{"action": "answer", "thought": "<error>", "content": "<original>"}`
- Fixed: trailing comma bug in regex (`TypeError: unterminated character set`)
- Enhanced for ReAct: passes through `thought` field, handles None/empty LLM responses
- Enhanced for Guard: defensive parsing for tool key in wrong location

#### tools.py (20719 bytes)
- 5 registered tools in structured `TOOLS` dict:
  - `python_code`: subprocess isolation, timeout, captures stdout/stderr
  - `read_file`: _resolve_safe_path prevents directory traversal
  - `write_file`: _resolve_safe_path, writes text
  - `calculator`: safe eval with operator validation
  - `search_kb`: wrapper around knowledge_base.py, lazy import
- Each tool has: fn, description, params (dict with type/required/description per param)
- `generate_tools_instruction()`: renders structured schemas into system prompt
- `execute_tool(name, params)`: central dispatch with error handling
- Import hygiene: no unnecessary imports

#### knowledge_base.py (7434 bytes)
- 23 facts across 5 domains (science, history, geography, technology, math)
- `get_fact(key)`: exact lookup
- `search_kb(query)`: 5-stage algorithm (exact match, keyword search, relevance scoring, fallback to value text, sorted by score)
- Fixed: "Kantō region" → "Kanto region in eastern Honshu island" (UnicodeEncodeError on Windows)

### Advanced Patterns (Notebooks 06-09)

#### guard.py (40448 bytes)
- **Layer 1: Name Resolution** (3-stage cascade)
  - Normalized exact match (O(1), lowercase + strip)
  - Prefix match (O(k), for partial names)
  - Levenshtein fallback (O(m*n), DP algorithm, character-level fuzzy matching)
- **Layer 2: Schema Validation**
  - `_check_type(param, expected_type)`: type validation
  - `validate_params(params, schema)`: required fields, type checking, error collection
- **Layer 3: Semantic Recovery**
  - `_try_coerce_to_boolean`: "true"/"false"/"yes"/"no" → bool
  - `_try_coerce_to_number`: "5"/"3.14"/"1.5e-10" → int/float
  - `_try_coerce_to_string`: anything → string representation
  - `_recover_value(key, value, schema)`: per-field recovery with type hint
  - `recover_params(params, schema)`: full parameter recovery with correction tracker
- `guarded_execute_tool(name, params)`: full 3-layer pipeline + recovery feedback loop
- 75+ comprehensive unit test suite (all pass)

#### react_agent.py (15289 bytes)
- `ReactCycle` @dataclass: thought, action, tool_name, tool_args, observation
- `ReActState` @dataclass: cycles list, current_cycle, is_complete, final_answer
- `build_react_system_prompt`: 3-action schema (use_tool, answer, think), mandates JSON
- `ReActAgent` class: step() (LLM call → parse → execute → add cycle), run() (loop until complete or max_steps)
- Handles pure "think" actions without creating full cycles
- Fixed: ASCII-safe console output for Windows
- Validated: 4-step, 3-cycle trace showing self-correction

#### plan_agent.py (10577 bytes)
- `PlanStep` @dataclass: id, description, dependencies, status, result
- `Plan` @dataclass: steps list, get_ready_steps (dependency resolution)
- `build_planner_prompt`: instructs LLM to produce ordered, dependency-aware plan as JSON
- `_extract_json()`: direct JSON extraction for planner output, defensive None handling
- `create_plan(task)`: LLM call to generate full plan
- `PlanAndExecuteAgent`: run method that resolves dependencies, executes sub-tasks via ReAct sub-agents, synthesizes final answer
- max_steps_per_subtask=8 (tuned for focused sub-task execution)
- Validated: 4 steps (search speed of light, search capital of France, calculate, write report), 12 total steps, 105s

#### reflection_agent.py (17259 bytes)
- `Critique` @dataclass: category, issue, suggestion
- `CritiqueReport` @dataclass: overall_score, dimension_scores dict, critiques list, summary
- `build_critic_prompt`: natural-language-optimized, 4 dimensions (accuracy, completeness, clarity, structure)
- `_extract_json() + _parse_natural_response()`: robust fallback for non-JSON outputs
- `build_revision_prompt`: specific issues + original answer → rewrite instruction
- `revise_answer`: temperature=0.7 for creative rewriting
- `ReflectionAgent` wrapper: accepts any inner_agent with .run(goal) method (Strategy/Decorator pattern)
- Loop: generate → critique → if score < 7.0 then revise → re-critique → pass/fail
- Fixed: critical build_revision_prompt bug (premature return/broken string concat)
- Validated: 6 fast tests pass, 4 LLM tests pass, 5.0→10.0 trajectory in 2 iterations

#### tree_of_thought.py (28657 bytes)
- `ThoughtNode` @dataclass: id, parent, depth, content, score, children, get_path, path_text
- `TreeOfThought`: branching_factor=3, max_depth=3, score_threshold=3.0, top_k=2
- `generate_thoughts`: parallel branch creation, temperature=0.8, concrete action-oriented prompt
- `evaluate_thought`: deterministic rubric scoring, temperature=0.3
- `bfs_search`: level-by-level, two-stage pruning (threshold + top_k), early termination on 10.0
- `dfs_search`: recursive deep dive, faster when good solutions are deep
- Critical fixes: fallback to `response.choices[0].message.reasoning` when content is None, max_tokens 1024/512
- Validated: BFS (8 calls, 17 nodes, early termination depth 2), DFS (4 calls, 4 nodes, 10.0 at depth 1)

#### iterative_refinement.py (29178 bytes)
- `FeedbackResult` @dataclass: score, passed, feedback_text, details, summary()
- `RefinementIteration` @dataclass: iteration, output, feedback, is_final
- `RefinementTrace` @dataclass: iterations, score_trajectory(), summary, total_time
- `IterativeRefinementAgent`: pluggable feedback_fn Callable[[str, str] -> FeedbackResult]
- 3-stage convergence: score threshold (>=8.0), plateau (improvement <0.3), decline (drop >0.5)
- 3 use cases:
  - safe_exec: subprocess sandbox, strips markdown before execution
  - code_test_feedback: dynamic test harness, counts PASS/FAIL
  - style_feedback: deterministic (word count, sentence length, passive voice, filler words)
  - fact_check_feedback: LLM claim extraction, DB lookup, verification
- extract_code_only: 4-stage fallback (markdown → bare def → full text → failsafe)
- Fixed: test harness string interpolation (concatenation instead of .format braces)
- Validated: 5 fast tests pass, 1 LLM test 10.0/10 on first attempt, temporary fix scripts deleted

### Memory Management (Notebook 10)

#### memory.py (27801 bytes)
- **Strategy pattern (GoF):** `MemoryStrategy` abstract base with 3 methods (`add`, `get_context`, `stats`)
- `Message` @dataclass: typed message wrapper (role, content, timestamp, lazy _token_count), replaces raw dicts
- **Strategy 1 -- `FullHistoryMemory`:** Keeps everything. Perfect recall, unbounded cost. Baseline for comparison.
- **Strategy 2 -- `SlidingWindowMemory`:** Keeps last N messages. Predictable budget, hard cutoff risk (system prompt can be evicted).
- **Strategy 3 -- `SummarizingMemory`:** Compresses old messages into running summary. Two modes: heuristic (fast, free, lossy) and LLM (accurate, costs API calls). Pluggable `llm_fn` via dependency injection.
- **Strategy 4 -- `ImportanceWeightedMemory`:** Scores messages by 4 factors (recency 0.3, length 0.2, keywords 0.3, role 0.2). Keeps messages above threshold, evicts lowest when max hit.
- `MemoryManager` unified controller: wraps any strategy, auto-migrates on `switch_strategy()` (replays full internal log), exposes `get_context_dicts()` for OpenAI API compat
- `STRATEGY_MAP`: data-driven factory dict -- adding a 5th strategy requires one class + one dict entry
- Backward-compatible wrappers: `build_prompt()`, `trim_for_api()` -- existing agent_loop.py works without changes
- 30-turn stress test: measures all 4 strategies on message count, token usage, and speed
- Validated: all 4 strategies initialize correctly, strategy switching migrates messages, backward compat with agent_loop.py confirmed

#### agent_loop.py (updated -- ~13000 bytes)
- Replaced `from memory import build_prompt` with `from memory import MemoryManager, Message`
- `__init__` accepts `memory_strategy` parameter (default "sliding") and `**memory_kwargs`
- Creates `MemoryManager` and seeds it with initial system + user messages
- `step()` uses `self.memory.get_context_dicts()` instead of `build_prompt()`
- Every action handler (think/answer/use_tool) calls `self.memory.add_dict()` to track new messages
- New methods: `get_memory_stats()` (diagnostic dict), `switch_memory_strategy()` (mid-run strategy swap)
- Finish/stop output now prints memory strategy, context message count, and estimated tokens

### Production Tool Design (Notebook 13)

#### tool_registry.py (~600 lines)
- `ToolResult` dataclass: standardized output with success/result/error/tool_name/execution_time, to_dict() for JSON serialization
- `ToolRegistry`: central hub with register(), validate_input(), call() with timing and error handling, discover(), get_stats(), per_tool_stats(), call_history audit log
- `validate_with_helpful_errors()`: LLM-friendly validation — returns None on success or formatted error string with what went wrong, what was expected, and how to fix it
- `ToolTestHarness`: automated testing framework with ToolTestCase (happy path, edge, error cases), run_suite() with clear table output
- **Structured error dict detection**: registry detects `{"success": False, "error": "..."}` dicts from tools and maps them to `ToolResult(success=False, error=...)`
- `StatefulKeyValueStore`: persistent key-value store (set/get/delete/list_keys) with access log — demonstrates stateful tool pattern
- `ConfirmationTool`: two-phase commit pattern (request_delete → confirm_action) for destructive operations — prevents accidental data loss
- `composed_store_and_verify()`: composed tool that chains kv_set + kv_get internally — demonstrates building complex workflows from simple tools
- `build_production_registry()`: integration point wiring all 20 tools together (5 base + 5 complex + 10 production)

#### production_tools.py (~550 lines)
- 10 production tools covering 5 categories: math (calculator, math_advanced), text (string_utils, text_stats), data (list_ops, dict_ops, format_converter, data_validator), utility (date_time, encoding_tools)
- Each tool follows 5 design principles: single responsibility, clear schemas, predictable errors, usage examples, fail gracefully
- `register_production_tools(registry)`: bulk registration function for all 10 tools
- `run_production_tools_tests()`: 24-test comprehensive suite (all pass)
- Tools covered: safe calculator (eval sandbox), string operations (upper/lower/title/reverse/etc.), list operations (sort/unique/frequencies/etc.), dict operations (keys/values/get/invert/etc.), date/time (now/parse/add_days/diff), text statistics (word count, sentence count, lexical diversity), format converter (JSON↔CSV↔markdown table), data validator (required/type/min/max/pattern/allowed checks), advanced math (mean/median/stdev/variance/gcd/lcm/percentiles), encoding (base64/url/md5/sha256/char_codes)

#### agent_tool_integration.py (~200 lines)
- `AdvancedToolAgent`: ReAct-style agent with tool registry integration
- System prompt built from registry.discover() — dynamically includes all registered tool descriptions
- JSON extraction with 4-fallback cascade (direct parse → markdown block → brace match → keyword scan)
- Agent loop: LLM call → parse → tool execution via registry → feed result back → repeat until answer or max_steps
- 3 demo queries validated with real LLM calls (calculator, file I/O, KB search)

#### test_production_llm.py (~170 lines)
- 10 real LLM integration tests exercising all production tools through the AdvancedToolAgent
- Each test has named queries with expected output validation
- Results: 10/10 PASS with real vLLM (Hermes) calls, 60.4s total wall time
- Notable: date_parse required self-correction (3 tool calls), math_stats demonstrated tool chaining (2 calls)

### Code Execution (Notebook 14)

#### static_analyzer.py (~530 lines)
- `SecurityAnalyzer` (AST NodeVisitor): walks Python AST to detect dangerous patterns
- Import checking: ALLOWED_MODULES whitelist (30+ safe modules), BANNED_MODULES blocklist (25+ dangerous modules)
- Dangerous builtin detection: exec, eval, compile, __import__, open, input, etc.
- Suspicious string patterns: command injection, hardcoded credentials, file destruction
- Complexity limits: max code length (50K chars), nesting depth (15), function count (50), loop count (20), literal size (10K)
- `analyze_code()`: full AST analysis → AnalysisResult with pass/fail, issue list, complexity score (0-100)
- `quick_check()`: fast pre-check with regex before expensive AST parse
- Validated: 7/7 tests pass (safe code, banned import, dangerous builtin, suspicious string, quick check, complexity scoring)

#### output_sanitizer.py (~360 lines)
- Sensitive pattern detection: API keys, tokens, secrets, AWS keys, private keys, passwords, emails, credit cards, SSNs, file paths
- Output truncation: max total length (1MB), max single line (10K), null byte stripping
- Content-type detection: JSON vs text vs error (with fallback parsing)
- `sanitize_output()`: full sanitization pipeline → SanitizedOutput with report
- `format_for_llm()`: ASCII-safe output formatting with truncation/redaction notices
- `format_error_for_llm()`: error formatting with helpful hints for self-correction (10+ error types)
- Validated: 7/7 tests pass (normal, JSON, error, truncation, redaction, LLM format, error hints)

#### audit_logger.py (~310 lines)
- `AuditEntry`: immutable dataclass with auto-generated timestamp, execution ID (SHA-256 based), full execution metadata
- `AuditLogger`: circular buffer (10K entries max), per-user filtering, statistics computation, JSON export
- Anomaly detection: alerts on N consecutive failures for same user
- `RateLimiter`: sliding-window per-user rate limiting, configurable max calls/window
- Validated: 5/5 tests pass (logging, rate limiting, user isolation, usage stats, export)

#### code_sandbox.py (~700 lines)
- `SubprocessSandbox`: production subprocess isolation with hard timeout enforcement (SIGKILL), stripped environment (no PYTHONPATH, no HOME, minimal PATH), temp file cleanup
- `DockerSandbox` (production-hardened): full container isolation with:
  - **seccomp profile** auto-discovery (`seccomp-profile.json`)
  - **cgroups v2**: memory limits + swap prevention, CPU quotas, PID limits (100)
  - **Capabilities**: `--cap-drop ALL`
  - **Network**: `--network none`
  - **Filesystem**: `--read-only` root + tmpfs `/tmp` and `/var/tmp`
  - **Security opts**: `no-new-privileges`, seccomp profile
  - **User**: non-root (UID 1000:1000)
  - **Environment**: PYTHONUNBUFFERED, PYTHONDONTWRITEBYTECODE, PYTHONHASHSEED=random, HOME=/tmp
  - **Monitoring**: background thread tracking docker stats (memory, CPU, PIDs), OOM detection
  - **Build helper**: `build_image()` with Dockerfile, `check_image_exists()` for validation
- `create_sandbox()`: factory function supporting both backends with same interface
- Validated: 7/7 tests pass (simple exec, math, error handling, timeout, env isolation, stats, factory)

#### code_executor.py (~400 lines)
- `CodeExecutor`: 5-layer pipeline (rate limit → quick check → AST analysis → sandbox → sanitization → audit)
- `CodeExecutionResult`: unified result with analysis, sandbox, sanitization, and audit metadata
- `execute_with_retry()`: automatic retry on failure with error context appended as comments
- `get_stats()`: comprehensive statistics (success rate, timing, per-layer blocking counts)
- `get_history()`: recent execution history with full metadata
- Validated: 8/8 tests pass (safe computation, blocked import, syntax error, runtime error, timeout, suspicious string, stats, history)

#### code_agent.py (~340 lines)
- `CodeAgent`: ReAct-style code generation with iterative self-correction
- `CodeCycle`: tracks thought, code, result per iteration
- `CodeAgentState`: full session state with cycle history, success tracking, timing
- System prompt: restricted module whitelist, code quality rules, JSON response format
- `_parse_response()`: 4-strategy JSON fallback (direct parse → markdown extraction → brace matching → failsafe)
- `_extract_answer()`: cleans output for final answer presentation
- `compare_code_vs_direct()`: head-to-head comparison of code agent vs direct LLM reasoning
- CodeAgent architecture correct, LLM integration validated (endpoint currently unavailable for live test)

#### Production Docker Files
- **`sandbox.Dockerfile`** (~90 lines): Hardened Alpine-based Python image. Removes curl/wget/git/gcc, installs only whitelisted packages (numpy, pandas frozen versions), removes pip entirely, creates non-root user (UID 1000), sets `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`, `PYTHONHASHSEED=random`. Health check included.
- **`seccomp-profile.json`** (~320 lines): seccomp BPF filter with ~350 allowed syscalls. Blocks: ptrace, process_vm_writev/readv, mount/umount/pivot_root, open_by_handle_at, bpf, perf_event_open. Default action: `SCMP_ACT_ERRNO` (block everything not explicitly allowed).
- **`docker-compose.sandbox.yml`** (~80 lines): Full orchestration with resource limits (256M memory, 1.0 CPU), security options, network isolation, read-only filesystem, tmpfs mounts, non-root user, health checks, logging limits, optional monitoring sidecar (prom/node-exporter).
- **`build_sandbox.ps1`** (~180 lines): Windows PowerShell build script. Validates Docker installation, checks required files, builds image, runs security audit (checks for dangerous binaries, verifies non-root user, confirms pip removal), runs functional tests (basic execution, network isolation, filesystem read-only).
- **`test_sandbox_security.py`** (~220 lines): Comprehensive security test suite. Tests SubprocessSandbox isolation (timeout, environment), tests full CodeExecutor pipeline (banned imports, dangerous builtins, suspicious strings), tests DockerSandbox (network, filesystem, memory limits, PID limits), tests resource governance (memory, CPU).
- **`sandbox_monitor.py`** (~180 lines): Real-time Docker container monitoring. Tracks memory usage, CPU percentage, PID count per container. Anomaly detection: alerts on >90% memory, >90% CPU, >80 PIDs, OOM kills. Parses `docker stats` output in background thread.
- **Test files**: `test_basic.py`, `test_network.py`, `test_write.py` — functional tests for Docker build script

### Web & Search Tools (Notebook 15)

#### web_search_tools.py (~260 lines)
- Tavily API client with production-grade error handling, request/response dataclasses
- `WebSearchResult` @dataclass: title, url, content, raw_content, score, published_date
- `WebSearchResponse` @dataclass: results list, search_time, total_tokens, answer (optional)
- `tavily_search()`: configurable max_results, search_depth (basic/advanced), include_answer, include_raw_content
- `register_tavily_tools()`: ToolRegistry integration — `web_search` tool with compact parameter schema
- **Compact output for ReAct context**: max 5 results, 500 chars each, no raw_content to prevent 65K context overflow
- Config-driven API key via `config.py` + `TAVILY_API_KEY` env var fallback
- Validated: smoke test + live search (5 results, ~5s, relevance 0.65-0.95)

#### tavily_research_agent.py (~150 lines)
- Direct search → synthesize pipeline (no ReAct loop)
- `TavilyResearchAgent`: `research(query)` method calls `tavily_search` → feeds results to LLM → returns synthesized answer with citations
- LLM prompt: strips URLs, keeps titles + snippets, asks for structured summary
- Validated: vague query "latest AI breakthroughs" → 5 results → clean synthesis in ~5s

#### demo_tavily_agent.py (~280 lines)
- `WebEnabledToolAgent`: ReAct-style agent subclass with web_search in ToolRegistry
- **Tuned system prompt**: "CRITICAL RULES" block forces synthesis after 1-2 searches (prevents infinite search loops)
- **Multi-object JSON scanner**: `_extract_json` scans all `{…}` blocks, picks the one with `tool` or `answer` key (handles hermes model outputting multiple JSON objects)
- **Context overflow fix**: web_search output compacted (5 results × 500 chars) to fit within 65,536 token limit
- Validated: 1 web_search call → immediate synthesized answer about April 2026 AI releases

#### multi_query_tavily_agent.py (~220 lines)
- `MultiQueryTavilyAgent`: LLM-driven query reformulation → parallel Tavily searches → URL-deduplicated merge → ranked results
- `QueryReformulationResult` @dataclass: original_query, reformulated_queries list
- `MergedSearchResult` @dataclass: title, url, sources (list of query IDs), combined_score, content
- Deduplication: URL normalization (strip fragments, lowercase, trailing slash), exact match + containment check
- Score merge: max score across duplicate sources, with source provenance tracking
- Validated: vague query (relevance 0.650) → 3 reformulated queries → top result 0.999, total ~30s

#### notebook15_corpus.py (~19KB, 30 documents)
- Complete local knowledge base from Notebook 15 curriculum materials
- 30 documents across 10 domains: AI Agents, AI Architecture, Machine Learning, Climate, Medical, Security, Physics, Economics, Biology, History
- Each document: `topic`, `title`, `content` (~80-120 words, condensed from originals)
- Used as local corpus for TF-IDF, semantic search, and hybrid retrieval benchmarks

#### local_search.py (~200 lines)
- Pure-Python TF-IDF search engine from scratch (matches Notebook 15 pedagogy)
- `LocalTFIDFSearchEngine`: manual tokenization (lower + strip + split), document indexing, vocabulary building
- `index()`: builds `doc_tokens` (token lists), `doc_freqs` (DF counts), computes `num_docs` and `avg_doc_length`
- `search()`: term frequency counting, IDF computation, score = Σ(tf × idf), results sorted descending
- **No external dependencies** (numpy for math operations only)
- Validated: 30 docs, 1,175 terms, instant search

#### semantic_local_search.py (~190 lines)
- Dense semantic search over Notebook 15 corpus using **fastembed + FAISS**
- **Auto-detects BGE availability**: uses `fastembed.TextEmbedding("BAAI/bge-small-en-v1.5")` when available, falls back to lightweight TF-IDF cosine if not
- **fastembed path**: ONNX Runtime (no PyTorch), 384-dim embeddings, FAISS IndexFlatIP for cosine similarity via inner product of L2-normalized vectors
- **Fallback path**: L2-normalized TF-IDF vectors + numpy dot product (cosine similarity)
- Lazy model loading (cached at module level)
- Validated: 30 docs embedded in ~1-2s, "The Transformer Architecture Revolution" surfaced at #3 for conceptual paraphrase query (proves semantic matching beyond keywords)

#### hybrid_search.py (~280 lines)
- `HybridSearchAgent`: three-source fusion — Tavily web + local TF-IDF + local BGE semantic
- `HybridSearchResult` @dataclass: source, title, url, content_preview, score (fused), raw_score
- Configurable weights: `web_weight`, `tfidf_weight`, `semantic_weight` (default 0.4/0.2/0.4)
- Score normalization: min-max for TF-IDF (arbitrary scale), semantic scores already [0,1] from cosine, Tavily scores already [0,1]
- Fusion formula: `fused = web_w × web_score + tfidf_w × tfidf_norm + semantic_w × semantic_score`
- Merged ranking: all sources combined, sorted by fused score descending
- Validated: conceptual paraphrase query shows web results dominating for current info, semantic surfacing foundational docs (transformer architecture), TF-IDF limited to keyword matches

### File & Data Tools (Notebook 16 — with OpenAI Data Agent Insights)

#### schema_aware_fs.py (~600 lines)
- **SchemaAwareFS**: dict-based in-memory filesystem with **6-layer metadata** inspired by OpenAI's data agent context architecture
  - **Layer 1 — Usage tracking**: `record_usage()` logs every operation with columns accessed, result summaries, access counts
  - **Layer 2 — Human annotations**: `annotate()` for curated descriptions, caveats, business meaning
  - **Layer 3 — Auto-inferred schema**: `_infer_csv_schema()` + `_infer_json_schema()` detect column types (int/float/bool/str), nullability, uniqueness, min/max/mean for numeric columns — all from sample data at create time
  - **Layer 3 — Data lineage**: `derive()` records full derivation chain (operation, source, parameters, timestamp); `get_lineage()` answers "where did this file come from?"
  - **Layer 5 — Memory integration**: `record_usage()` + `get_popular_columns()` feed learned corrections into the agent's memory system
  - **Layer 6 — Runtime validation**: `get_lineage()` + `stats()` provide live filesystem introspection
- Content type auto-detection (csv/json/text) via `_detect_content_type()`
- Schema staleness tracking after `append()` operations
- **Validated**: 8/8 tests pass (CSV schema inference, JSON schema inference, read with metadata, lineage tracking, popular columns, error handling, global stats, schema staleness)

#### unified_data_tools.py (~850 lines)
- **OpenAI Lesson #1 — "Less is More"**: consolidated 6 separate tool classes (CSVTools, JSONTools, StatisticsTools) into **4 unified interfaces**
  - `DataReader`: parses CSV with **auto-type detection** (int/float/bool/str), parses JSON — returns structured dicts with schema metadata
  - `DataQuery`: **unified filter/sort/group/aggregate/projection** in one call — SQL-like conditions list, multi-column sort, group-by with multiple aggregates, column selection, limit
  - `DataStats`: **comprehensive statistics with auto-type detection** — numeric summaries (mean, median, std, percentiles, IQR) or categorical (frequency counts), plus correlation + linear regression
  - `DataTransform`: `derive()` (lambda or string expression), `select()`, `rename()`, `pivot()` (long-to-wide with configurable aggregation)
- All methods accept filesystem results or raw data for composability
- All methods return structured `{"success", "error", ...}` dicts (never exceptions)
- **Validated**: 10/10 tests pass (auto-typed CSV read, JSON read, unified query, group-by with aggregates, auto-detect stats + categorical, correlation, regression, derive, rename, pivot)

#### self_correcting_data_agent.py (~750 lines)
- **OpenAI Lesson #2 — "Guide the Goal, Not the Path"**: LLM plans high-level analysis steps; deterministic tools execute precisely
- **Self-correcting agent architecture**: Plan → Execute → Detect Anomaly → Self-Correct → Retry → Synthesize
  - `DataAnomaly` dataclass: type (zero_rows, null_heavy, type_mismatch, narrow_result), severity (warning/error/critical), suggestion
  - `AnalysisStep` dataclass: tracks tool, parameters, reasoning, anomalies, retry count, correction status
  - `AnalysisResult` dataclass: full session trace with step count, anomalies found, self-corrections, memory entries, execution time
- **Anomaly detection** (`_detect_anomalies()`):
  - Zero rows after filter → too restrictive
  - >50% null in any column → data quality issue
  - Type mismatch in stats (e.g., non-numeric column requested)
  - Single-row result without explicit limit → possible over-filtering
- **Self-correction strategies** (`_self_correct()`):
  - Zero rows: remove conditions one at a time, or change `==` to `contains`
  - Type mismatch: switch to auto-detect numeric columns
  - All corrections track retry count; max_retries configurable (default 2)
- **Memory integration**: stores learned corrections per dataset (`_learned` dict); recalls hints before planning; compatible with `LongTermMemory` from Notebook 11
- **Workflow templates** (`WorkflowTemplate` dataclass):
  - Built-in: `explore` (read + stats), `compare_groups` (group-by + sort), `correlation_map` (read + stats + pairwise correlations)
  - Custom registration via `register_workflow()` with parameter substitution (`{group_col}` → actual value)
- **Clarifying questions**: `ask_clarifying_question()` detects vague queries ("tell me about the data") and asks for specifics — or applies sensible defaults
- **LLM timeout handling**: planning/synthesis `timeout=180`, clarifying `timeout=300` — reasoning models (Hermes/vLLM via Modal) need time for thinking phase before producing content
- **Reasoning-model awareness**: handles `message.content=None` by falling back to `message.reasoning` field (characteristic of thinking models like o1/DeepSeek-R1)
- **Validated**: 10/10 deterministic tests pass; **1 live LLM test passed** — endpoint warm, 84.9s total, 3-step plan (query→stats→synthesize), zero anomalies, substantive synthesized answer

#### demo_data_agent.py (~500 lines)
- Full end-to-end demonstration with **countries dataset** (20 countries, 5 columns)
- **SchemaAwareFS setup**: auto-inferred schema, human annotations, popular column tracking
- **Unified tools demo**: Asian countries query (filter + sort + limit), GDP statistics, population-GDP correlation, derive GDP per capita, save with lineage
- **Anomaly demo**: `continent == Antarctica` → zero rows → would trigger self-correction
- **Workflow execution**: built-in `explore` + custom `gdp_per_capita_ranking` workflow
- **Evaluation cases** (golden dataset with expected answers):
  - `asia_gdp_total`: expects "$29,400 billion" — **PASS**
  - `top_gdp_per_capita`: expects "United States" + "GDP per capita" — **PASS**
  - `continent_comparison`: expects continent names + "average GDP" — **PASS**
  - `population_density`: expects "Bangladesh, South Korea, India" + "density" — **PASS**
  - `pop_gdp_correlation`: expects "correlation" + "0." — **PASS**
  - **Result: 5/5 evaluation cases pass**
- **LLM integration**: endpoint connectivity check (HEAD request, 3s timeout); when available, full `agent.analyze()` with planning + synthesis; when unavailable, deterministic fallback traces
- **Live LLM test**: Successfully executed on warm Modal endpoint (Hermes/vLLM reasoning model). 84.9s total: planning (55s reasoning + JSON output), execution (instant), synthesis (30s). 3-step plan produced, zero anomalies, 1,129-char substantive answer
- **Endpoint-aware**: gracefully degrades to deterministic output when LLM is unreachable

### Diagnostic & Documentation

- `config.py` (967 bytes): LLM_CONFIG dict, singleton OpenAI client, get_model() helper, vLLM endpoint
- `test_reflection_fast.py` (10267 bytes): 6 fast unit tests for reflection data structures and prompt construction
- `test_reflection_llm.py` (10534 bytes): 4 LLM validation tests (critic flags, reviser fixes, re-evaluation passes, full loop)
- `comparison_test.py` (4155 bytes): ReAct vs simple loop head-to-head
- `debug_basic.py`, `debug_critic.py`, `debug_critic2.py`: diagnostic scripts for isolating None LLM responses
- `REACT_SUMMARY.md` (2836 bytes), `PLAN_AND_EXECUTE_SUMMARY.md` (4060 bytes), `COMPARISON.md` (6752 bytes), `REFLECTION_WALKTHROUGH.md` (10548 bytes): architecture documentation

---

## Part 3: Where We Are — The Current State

### ✅ COMPLETED (15 of 37 notebooks fully covered)

| Notebook | Custom Build Equivalent | Validation |
|----------|------------------------|------------|
| 01 - Introduction | Concepts understood, no separate file | N/A |
| 02 - Agent Loop | state.py + agent_loop.py (core) | Live 3-step workflow test |
| 03 - Tool Use | tools.py (5 tools, structured schemas) | 6-step multi-tool workflow |
| 04 - Structured Parsing | parser.py (4 strategies) + guard.py (75+ tests) | All 75+ tests pass |
| 05 - ReAct | react_agent.py + knowledge_base.py | 4-step, 3-cycle, 16.8s |
| 06 - Plan-and-Execute | plan_agent.py | 12 steps, 105s, 4 sub-tasks |
| 07 - Reflection | reflection_agent.py + test suites | 5.0→10.0 trajectory |
| 08 - Tree of Thought | tree_of_thought.py | BFS 8 calls, DFS 4 calls |
| 09 - Iterative Refinement | iterative_refinement.py | 5 fast + 1 LLM test, 10.0/10 |
| **10 - Short-Term Memory** | **memory.py (4 strategies) + agent_loop.py integration** | **4 strategies init OK, switch OK, 30-turn stress test passed** |
| **11 - Long-Term Memory** | **long_term_memory.py (BGE 384-dim, FAISS, episodic + semantic, persistence, decay)** | **Demo: 5 episodes + 5 facts, recall scores 0.74-0.87, save/load OK, decay OK** |
| **12 - Knowledge Graph Memory** | **graph_memory.py (GraphMemory with nx.DiGraph, entity canonicalization, BFS multi-hop, LLMQueryClassifier, HybridMemory, GraphMaintenance)** | **Triples extraction, multi-hop queries, vector+graph hybrid, contradiction detection, health reports** |
| **13 - Advanced Tool Design** | **tool_registry.py (ToolRegistry, ToolResult, validate_with_helpful_errors, ToolTestHarness, StatefulKeyValueStore, ConfirmationTool, composed tools) + production_tools.py (10 tools: calculator, string_utils, list_ops, dict_ops, date_time, text_stats, format_converter, data_validator, math_advanced, encoding_tools) + agent_tool_integration.py (AdvancedToolAgent) + test_production_llm.py (real LLM integration tests)** | **24/24 unit tests pass, 13/13 registry tests pass, 10/10 production tool LLM tests pass, 3/3 base agent demos pass** |
| **14 - Code Execution Tool** | **static_analyzer.py + output_sanitizer.py + audit_logger.py + code_sandbox.py (SubprocessSandbox hard-kill + production DockerSandbox with cgroups, cap-drop, read-only, tmpfs, PID limits, monitoring) + code_executor.py + code_agent.py + sandbox.Dockerfile + seccomp-profile.json + docker-compose.sandbox.yml + build_sandbox.ps1 + sandbox_monitor.py + test_sandbox_security.py** | **52/52 tests pass (7+7+5+7+8+18 security), DockerSandbox FULLY VERIFIED with live container execution: network blocked, read-only enforced, OOM kills 300MB, fork bomb blocked, non-root user confirmed** |
| **15 - Web and Search Tools** | **web_search_tools.py (Tavily API client, dataclasses, registry integration, compact output) + tavily_research_agent.py (direct search→synthesize) + demo_tavily_agent.py (ReAct with tuned prompt, multi-object JSON scanner, context overflow fix) + multi_query_tavily_agent.py (LLM query reformulation, URL deduplication) + notebook15_corpus.py (30-doc local corpus) + local_search.py (pure-Python TF-IDF from scratch) + semantic_local_search.py (fastembed BGE + FAISS, auto-fallback) + hybrid_search.py (3-source fusion: web 0.4 + TF-IDF 0.2 + semantic 0.4)** | **Tavily: smoke test + live search (5 results, ~5s, 0.65-0.95 relevance); ReAct demo: 1 search → immediate synthesis (prevents infinite loop); Multi-query: 3 reformulations → merged top result 0.999; Hybrid: conceptual paraphrase query shows TF-IDF limited to keywords, BGE semantic surfaces transformer architecture doc, web fills current gaps** |

### ⬜ REMAINING (21 notebooks ahead)

**Module 3 (Tool Engineering) is COMPLETE.** The next target in strict curriculum order is **Notebook 17: Multi-Agent Conversation**.

The full remaining curriculum spans 4 modules:

1. **Module 4**: Notebooks 17-23 (Multi-Agent: conversation, debate, pipelines, hierarchy, orchestration, blackboard, swarm)
2. **Module 5**: Notebooks 24-28 (Production: safety, human-in-loop, evaluation, cost, resilience)
3. **Module 6**: Notebooks 29-31 (Protocols: MCP, A2A, runtime)
4. **Module 7**: Notebooks 32-37 (Capstone: Castalia Scholar 6-part project)

---

## Part 4: Key Architectural Patterns Learned

These are the recurring patterns that will inform all future lessons:

### 1. Agent as a Loop
The foundational abstraction. Every agent (simple, ReAct, plan, reflection, iterative) is a while loop with:
- State (what's been done so far)
- Step function (one iteration of reasoning + action)
- Termination condition (when to stop)

### 2. Pluggable Components
Every major system accepts interchangeable parts:
- Parser: 4 strategies cascading, each independent
- Guard: 3 layers, each independent validation stage
- Reflection: accepts any inner agent (Strategy/Decorator pattern)
- Iterative: accepts any feedback function (Callable interface)

### 3. Structured Data Contracts
Every inter-component communication uses @dataclass:
- AgentState, ReactCycle, PlanStep, Critique, FeedbackResult, ThoughtNode
- This prevents silent data corruption and makes the system debuggable

### 4. Defensive Parsing
Never trust LLM output. Always:
- Try direct parse first
- Fall back to regex extraction
- Fall back to semantic recovery
- Fall back to failsafe defaults
- Feed errors back to LLM for self-correction

### 5. Progressive Complexity
Each module builds on the previous:
- Single prompt → Loop (adds state) → ReAct (adds trace) → Plan (adds decomposition) → Reflection (adds quality gate) → Iterative (adds external feedback)

### 6. Thinking Model Quirks
The hermes-model uses a `reasoning` field for chain-of-thought. When tokens are low, `content` may be None. Always:
- Check `content or response.choices[0].message.reasoning`
- Set adequate max_tokens (1024+ for reasoning-heavy prompts)
- This was a hard-won lesson across 3 separate modules

### 7. Strategy Pattern for Swappable Behavior
Memory management demonstrated the Strategy pattern in production:
- `MemoryStrategy` abstract base defines the interface (`add`, `get_context`, `stats`)
- 4 concrete strategies implement it independently
- `MemoryManager` holds one strategy at a time and delegates
- `switch_strategy()` migrates data without breaking the agent loop
- This same pattern appears in: Parser (4 strategies), Guard (3 layers), Reflection (pluggable inner agent), Iterative (pluggable feedback fn)

### 8. Short-Term vs Long-Term Memory Separation
Notebook 11 established that STM and LTM solve different problems:
- STM: "What messages should I send to the LLM right now?" (bounded, ephemeral)
- LTM: "What do I know across all sessions?" (unbounded, persistent)
- They are **complementary, not interchangeable** — an agent uses both
- STM uses strategy-swappable buffers; LTM uses vector-searchable stores (FAISS) with separate episodic/semantic divisions

### 9. Embeddings Enable Semantic Search
Moving from TF-IDF to BGE (BAAI/bge-small-en-v1.5) demonstrated the qualitative leap:
- TF-IDF: keyword overlap only ("light" in query matches "light" in text)
- BGE: captures semantic relationships ("sorting" ~ "bubble sort" ~ "Timsort", even with zero keyword overlap)
- 384-dimensional vectors replace 1024-dimensional sparse TF-IDF vectors
- FAISS IndexFlatIP does exact inner-product search (cosine similarity after L2 normalization)

### 10. Memory Maintenance Operations
Long-term memory requires active maintenance:
- **Decay**: exponential half-life formula `importance × 2^(-age/half_life)` prunes stale memories
- **Reinforcement**: re-storing a fact timestamps it and boosts confidence
- **Consolidation**: merges duplicate/similar facts to prevent knowledge bloat
- **Persistence**: JSON save/load enables cross-session recall

### 11. Knowledge Graphs Complement Vector Stores
Vectors and graphs solve fundamentally different query types:
- **Vectors** answer "find me something ABOUT X" via semantic similarity (cosine distance in embedding space)
- **Graphs** answer "how does X connect to Y?" via edge traversal (BFS/DFS on entity-relation-object triples)
- **Multi-hop reasoning** is the graph's killer feature: Alice → manages → Backend Team → owns → Payment Service → uses → PostgreSQL requires 3 traversals

### 12. Entity Canonicalization Prevents Graph Pollution
Without normalization, "Alice", "alice", and "ALICE" become 3 disconnected nodes storing different facts about the same person. The `entity_index` dict maps `normalized.lower().strip()` → `canonical_first_seen_form` so all variants resolve to one node. This is the graph equivalent of database deduplication.

### 13. LLM-Based Query Routing Replaces Keyword Heuristics
The notebook's hardcoded `RELATION_KEYWORDS` set is a pedagogical simplification, not production code. The actual implementation in `graph_memory.py` uses `LLMQueryClassifier` (Option B):
- **Pros**: Handles synonyms, new domains, multilingual queries, complex phrasing
- **Cons**: Adds 50-200ms latency + one LLM call per user query
- **Fallback**: `BOTH` default if LLM returns unexpected text (safe degradation)
- **Cache**: `_cache` dict prevents repeated classification of identical queries

### 14. Graph Maintenance Requires Active Monitoring
A knowledge graph without maintenance becomes a garbage pile:
- **Contradictions**: same (subject, relation) → different objects for unique relations (e.g., two CTOs)
- **Redundancy**: semantically identical triples from LLM extraction variation ("manages" vs "is manager of")
- **Health metrics**: connected components (should be ~1), isolated nodes (should be ~0), degree distribution (avg 2-5 is healthy)
- **O(n²) redundancy scan** via embedding similarity is acceptable for <1000 edges; FAISS ANN needed for larger graphs

### 15. Registry Pattern Centralizes Tool Management
Notebook 13 introduced `ToolRegistry` as a unified hub for tool dispatch, replacing ad-hoc dicts:
- **Registration**: tools paired with formal `ToolDefinition` schemas at register time
- **Validation**: `validate_input()` checks kwargs against `ParameterSchema` rules before execution
- **Dispatch**: `call()` handles validation → execution → timing → error handling in one pipeline
- **Discovery**: `discover()` renders all tool descriptions into LLM-readable system prompts
- **Audit**: `call_history` logs every call with args, result, and timing for debugging and stats

### 16. Structured Error Dicts Enable LLM Self-Correction
Production tools return `{"success": False, "error": "..."}` instead of raising exceptions:
- The registry detects these and maps them to `ToolResult(success=False, error=...)`
- Error messages are written as corrective prompts (what went wrong + how to fix it)
- The agent loop feeds errors back to the LLM, which can retry with corrected inputs
- This was validated in the real LLM tests: `date_parse` self-corrected from a wrong format string after receiving the error message

### 17. Complex Tool Patterns: Stateful, Confirmation, Composed
Three advanced tool types beyond simple function dispatch:
- **Stateful**: `StatefulKeyValueStore` maintains persistent state across calls (like a mini-database)
- **Confirmation**: `ConfirmationTool` implements two-phase commit — request returns a token, confirm executes the action (prevents accidental data loss)
- **Composed**: `store_and_verify` chains multiple registry calls internally, demonstrating how to build complex workflows from simple tools
- Key insight: composed tools do **not** auto-rollback on failure — they report step-wise results and let the agent decide recovery policy

### 18. Test Harness Decouples Tool Testing from Agent Testing
`ToolTestHarness` provides systematic testing separate from LLM calls:

### 19. Defense in Depth for Code Execution
Notebook 14 established that security requires multiple independent layers:
- **No single layer is sufficient**: regex misses obfuscated code, AST can't catch runtime behavior, subprocess isolation can't prevent bad logic
- **Stacked layers cover each other's blind spots**: if static analysis misses it, the sandbox kills it; if the sandbox times out, the audit log records it
- **Layer independence is critical**: each layer can block execution independently, so a bypass in one doesn't compromise the whole system
- **Production ≠ educational**: thread-based timeouts are cooperative (can't force-kill); subprocess timeouts are hard (SIGKILL); Docker containers add syscall filtering, memory limits, and network isolation

### 20. Pipeline Architecture for Tool Execution
The `CodeExecutor` demonstrates a production pipeline pattern:
- **Sequential stages with independent blocking**: rate limit → pre-check → AST analysis → sandbox → sanitization → audit
- **Each stage can independently fail**: early stages fail fast (cheap), later stages fail with more context (expensive)
- **Structured results at every stage**: every layer returns typed data (AnalysisResult, ExecutionResult, SanitizedOutput, AuditEntry)
- **Retry with context**: failed executions feed error messages back as comments for self-correction

### 21. AST-Based Analysis Beats Regex for Security
The `SecurityAnalyzer` (AST NodeVisitor) demonstrates why regex is insufficient for security:
- **Regex** can be bypassed by obfuscation (variable names, string concatenation, encoding)
- **AST** understands the actual Python syntax structure — `import os` is caught whether it's `import os`, `import os as o`, or `from os import system`
- **AST walks the entire tree**: it catches dangerous calls, suspicious strings, complexity limits, and resource exhaustion risks in a single pass
- **Complexity scoring** (0-100) provides a quantitative measure of code risk beyond just pass/fail
- `ToolTestCase`: defines tool name, inputs, expected success, expected substrings in result/error
- `run_suite()`: executes all tests against a registry and prints a clear pass/fail table
- Separation of concerns: unit tests (deterministic, fast) vs integration tests (real LLM, slower)
- Production tools: 24/24 unit tests + 10/10 real LLM integration tests

### 22. Search Tool Output Must Be Context-Budget Disciplined
ReAct agents feed tool output back into the LLM as new messages. Unbounded search results blow context windows:
- **Tavily advanced search** returns full page text (~10K-50K tokens per result × 10 results = 500K tokens)
- **With a 65,536 token limit**, this means ONE tool call consumes the entire budget
- **Solution**: compact tool results to max 5 results, 500 chars each, no `raw_content` — reduces output from ~500K to ~2.5K tokens
- The `web_search_tool` registered in the ToolRegistry enforces this compaction at the tool layer, not the agent layer
- **Key insight**: context budgeting is a tool design concern, not just a prompt engineering concern

### 23. "CRITICAL RULES" Prompt Blocks Override LLM Behavior Tendencies
The hermes model consistently treated "answer after 1-2 searches" as advisory, not mandatory, and entered infinite search loops:
- **Initial approach**: polite instructions in system prompt ("You should provide a final answer after searching")
- **Failure**: model interpreted this as optional guidance, treated step budget as "keep refining"
- **Fix**: **"CRITICAL RULES"** block with imperative language:
  - "You may call web_search at most TWO times"
  - "After your FIRST search, if results contain relevant information, you MUST immediately provide a final answer"
  - "If you call web_search more than TWO times, you have FAILED"
- **Result**: 1 search → immediate synthesis, 2 steps total (down from 6-8 steps of repeated searching)
- **General principle**: when an LLM has a behavioral bias (caution, repetition, over-tooling), override with extreme imperative language, not polite suggestions

### 24. Multi-Query Reformulation Compensates for LLM Vagueness
Vague queries like "latest AI breakthroughs" return poor results (relevance 0.650) because they lack specificity:
- **Strategy**: LLM reformulates one vague query into 3 specific queries ("latest AI model releases 2026", "recent AI research papers 2026", "new AI applications deployed 2026")
- **Parallel execution**: all 3 queries sent to Tavily simultaneously
- **Deduplication**: URL normalization (strip fragments, lowercase, trailing slash) + exact match + containment check prevents the same article appearing 3 times
- **Score merge**: max score across duplicate sources (a page that appears for 2 queries gets the higher of its two scores)
- **Result**: top result relevance improved from 0.650 → 0.999
- **Trade-off**: 3× API cost, but ~3× better coverage; viable when result quality matters more than cost

### 25. Three-Source Hybrid Search: Web + Sparse + Dense
No single retrieval method covers all query types:
- **Web (Tavily)**: current events, breaking news, post-training-cutoff information — scores naturally in [0,1]
- **Local TF-IDF**: keyword-matching over curated corpus — fast, interpretable, but fails on paraphrase — scores on arbitrary scale, require min-max normalization
- **Local Semantic (BGE/fastembed)**: dense embeddings capture conceptual similarity even without keyword overlap — scores in [0,1] as cosine similarity
- **Fusion formula**: `fused = w_web × web_score + w_tfidf × norm(tfidf) + w_semantic × semantic_score`
- **Default weights**: web=0.4, tfidf=0.2, semantic=0.4 — gives semantic local significant weight while keeping web current
- **Validation**: conceptual paraphrase query ("neural networks that read entire sentences at once") — TF-IDF fails to find transformer document, BGE semantic surfaces it at #3, web fills current gaps with Medium/Instagram/Reddit articles about transformers
- **Architecture**: auto-detects fastembed availability, falls back to lightweight TF-IDF cosine — same API regardless of backend

### 26. fastembed: ONNX-Based Embedding Without PyTorch
The `sentence-transformers` + `torch` stack is heavy (~2GB, requires compilation on some platforms). fastembed provides the same BGE models via ONNX Runtime:
- **API**: `TextEmbedding(model_name="BAAI/bge-small-en-v1.5")` → `.embed(texts)` returns generator of numpy arrays
- **Size**: ~100MB model file (ONNX), vs ~2GB PyTorch
- **Speed**: comparable inference speed, no GPU required
- **Compatibility**: same 384-dimensional vectors, same cosine similarity behavior after L2 normalization
- **FAISS integration**: identical — `IndexFlatIP` with normalized vectors = cosine similarity
- **Fallback**: if fastembed unavailable, lightweight TF-IDF cosine via numpy provides conceptually similar behavior (co-occurrence semantics) at zero install cost

### 27. Multi-Layer Metadata Enables Grounded Reasoning
Inspired by OpenAI's 6-layer data agent context system (table usage, human annotations, Codex enrichment, institutional knowledge, memory, runtime):
- **Schema inference** at file creation time captures column types, nullability, min/max/mean — all without pandas
- **Usage tracking** records which columns are queried most often, enabling the agent to prioritize relevant fields
- **Human annotations** store business meaning and caveats that raw schemas cannot capture
- **Data lineage** (`derive()` + `get_lineage()`) answers "where did this file come from?" — critical for debugging derived datasets
- **Schema staleness** tracking after `append()` prevents the agent from using outdated type information

### 28. Tool Consolidation Reduces Agent Confusion ("Less is More")
OpenAI Lesson #1: overlapping tools confuse agents. The original Notebook 16 had 6 separate tool classes (CSVTools, JSONTools, StatisticsTools) with 15+ methods:
- **Consolidated to 4 unified interfaces**: `DataReader` (parse), `DataQuery` (filter/sort/group/aggregate/projection), `DataStats` (summarize/correlate/regress), `DataTransform` (derive/select/rename/pivot)
- **Single entry point**: `DataReader.read()` auto-detects CSV vs JSON, returns structured result with schema
- **Chaining**: all tools accept either filesystem results or raw data, returning structured dicts — composable without type gymnastics
- **Result**: agent calls fewer tools with clearer intent, reducing ambiguity and improving reliability

### 29. Goal-Oriented Prompting Beats Prescriptive Instructions ("Guide the Goal, Not the Path")
OpenAI Lesson #2: highly prescriptive prompting degraded results by pushing the agent down incorrect paths:
- **System prompt design**: LLM receives dataset schema + available tool descriptions + high-level goal ("answer the question")
- **Agent plans**: LLM generates step list (tool + parameters + reasoning), not step-by-step imperative code
- **Tools execute**: deterministic computation with exact results (filter, sort, stats, transform)
- **Self-correction**: if intermediate result looks wrong (zero rows, null-heavy column), agent investigates and retries with adjusted parameters — without user intervention
- **Result**: robustness improves because the agent chooses the appropriate execution path, not a rigid script

### 30. Data Lineage Answers "Where Did This Come From?" ("Meaning Lives in Code")
OpenAI Lesson #3: pipeline logic captures assumptions, freshness guarantees, and business intent that never surface in metadata alone:
- **`derive()` method**: creates a derived file while recording full lineage — operation name, source path, parameters, timestamp
- **Derivation chain**: list of `{operation, source, parameters, timestamp}` — the agent can trace any file back to its origins
- **Cross-reference**: source file's `annotations["produced"]` records which derived files it generated
- **Use case**: when an agent sees `/data/countries_with_gdpc.csv` and asks "how was this computed?", `get_lineage()` returns `"derive_gdp_per_capita from /data/countries.csv with expression: gdp * 1000 / population"`

### 31. Self-Correction Shifts Iteration from User to Agent
OpenAI's data agent evaluates its own progress — if intermediate results look wrong, it investigates and adjusts:
- **Anomaly detection** (`_detect_anomalies()`): zero rows (too restrictive filter), null-heavy columns (>50% null), type mismatch (stats on categorical), narrow result (1 row without limit)
- **Self-correction strategies** (`_self_correct()`): remove conditions one at a time, change `==` to `contains`, switch to auto-detect numeric columns
- **Retry loop**: each step gets `max_retries` attempts (default 2); after each retry, re-detect anomalies
- **Memory of corrections**: successful corrections are stored per dataset (`_learned` dict) and recalled before future planning — preventing repeated mistakes
- **Result**: user asks once, agent iterates internally; converges faster and with higher quality than manual workflows

### 32. Evaluation Cases Are Unit Tests for Data Agents
Inspired by OpenAI's Evals API: curated question-answer pairs with "golden" expected outputs:
- **EvalCase dataclass**: name, question, expected_answer_type, expected_contains (substrings), tolerance (for numeric)
- **Deterministic answer builder**: `_deterministic_answer()` computes expected results using the same tools the agent uses — no LLM involvement, perfect reproducibility
- **Validation**: check that actual answer contains expected substrings (flexible — exact numbers may vary, key concepts must appear)
- **Result**: 5/5 evaluation cases pass on the countries dataset, confirming correctness of the unified tools
- **Future**: when LLM synthesis is added, eval cases become regression tests — same questions, verify synthesized answers contain same key facts

---

## Part 5: Domain-Tutor Handoff Instructions

### Resume Point
The learner has completed **Notebooks 01-16** (all 16 notebooks of Modules 1-3). The next target in strict curriculum order is **Notebook 17: Multi-Agent Conversation**.

### What the Tutor Should Do Next
1. **Acknowledge the progress**: The learner has built a remarkably comprehensive 16-module custom system spanning Foundations → Single-Agent Patterns → Memory → Tool Engineering. Each notebook concept was not just understood but implemented, tested, and hardened against real LLM behavior.

2. **Start with Notebook 17**: The next natural progression is multi-agent systems — two or more agents conversing, debating, and collaborating. This shifts from "one agent with many tools" to "many agents with distinct roles."

3. **Scaffolding level**: The learner has demonstrated mastery through Notebooks 01-16, including critical architectural decisions (LLMQueryClassifier over keyword routing, Docker sandbox over subprocess, fastembed over sentence-transformers, unified data tools over 6 separate classes). They are ready for pseudocode + key design decisions, then drive implementation.

4. **Pedagogy preference**:
   - Architecture diagrams first, then code
   - Emphasize trade-offs and design decisions
   - Active recall on how new concepts connect to prior modules
   - Project-based: extend the existing codebase

5. **Key concepts to emphasize in Notebook 17**:
   - Agent specialization: why dedicated agents outperform generalists (Researcher vs Skeptic vs Teacher vs Student)
   - Message passing: `Message` dataclass with sender, recipient, content, timestamp, metadata
   - Conversation mechanics: turn-taking, termination conditions, role-based response generation
   - State isolation vs shared state: each agent has its own memory, or they share a blackboard?
   - TwoAgentChat: the simplest multi-agent pattern — specialized roles in a structured dialogue

6. **Tooling note**: The `edit` tool frequently fails on complex multi-line replacements. The established workaround is:
   - For small fixes: `bash sed` or `bash python` one-liners
   - For large additions: full file rewrite using `write` or bash heredoc
   - This has been tested across 16 modules and is the reliable pattern

7. **Active recall from Notebook 12**: The hardcoded keyword routing question was answered with implementation of LLMQueryClassifier (Option B). The learner understands that keyword matching is a teaching crutch, not production code.

8. **Terminal encoding**: Windows console uses cp1252. Any Python scripts that print non-ASCII characters will crash. Always use ASCII or set `encoding='utf-8'` explicitly.

9. **Active recall from Notebook 13**: The learner built a full production tool registry with 20+ total tools (5 base + 10 production + 5 complex types + Tavily web search), a test harness with 100% pass rates, and validated all tools with real LLM calls. Key insight: tools return structured error dicts with `success: false` and `error` messages, and the registry detects these to report failures correctly.

10. **Active recall from Notebook 15**: The learner integrated live web search (Tavily) with local retrieval (TF-IDF + BGE semantic via fastembed) in a three-source hybrid architecture. Key insight: context budgeting is a tool design concern — the `web_search` tool compacts output (5 results × 500 chars) to prevent 65K context overflow. Also: "CRITICAL RULES" prompt blocks with imperative language override LLM behavioral biases (e.g., infinite search loops) where polite suggestions fail.

11. **Active recall from Notebook 16**: The learner built a production-grade data analysis system inspired by OpenAI's in-house data agent. Key insights: (a) "Less is More" — consolidating 6 separate tool classes into 4 unified interfaces reduced agent confusion; (b) "Guide the Goal, Not the Path" — LLM plans high-level, tools execute deterministically; (c) self-correction with anomaly detection shifts iteration from user to agent; (d) data lineage (`derive()` + `get_lineage()`) captures provenance that metadata alone cannot. 5/5 evaluation cases pass on countries dataset.

### Files the Tutor Can Reference
- `PROGRESS.md` (this file) — complete curriculum map
- `REFLECTION_WALKTHROUGH.md` — detailed architecture of the reflection system
- `COMPARISON.md` — side-by-side comparison of all 4 agent patterns
- `schema_aware_fs.py` — SchemaAwareFS with 6-layer metadata, schema inference, derivation chains
- `unified_data_tools.py` — Consolidated DataReader/DataQuery/DataStats/DataTransform (4 classes replacing 6)
- `self_correcting_data_agent.py` — Anomaly detection, self-correction, workflow templates, memory integration
- `demo_data_agent.py` — Countries dataset demo, 5/5 evaluation cases, LLM synthesis with timeout
- `build-my-agent/*.py` — the actual working code (25+ production-ready modules)

### What Comes After Notebook 16
**Module 3 (Tool Engineering) is COMPLETE.** The sequence continues:

**Module 4: Multi-Agent Systems**
- 17 (multi-agent conversation) — TwoAgentChat, specialized roles, message passing
- 18 (debate and consensus) — DebateArena, DebaterAgent, JudgeAgent, voting
- 19 (sequential pipelines) — AgentNode, Pipeline, error handling, partial execution
- 20 (hierarchical delegation) — Manager-worker pattern, decompose-delegate-aggregate
- 21 (orchestration patterns) — Router Agent, conditional routing, parallel fan-out/fan-in
- 22 (shared state/blackboard) — Shared workspace, event-driven, conflict resolution
- 23 (swarm intelligence) — Decentralized agents, stigmergy, emergent behavior

The full tool engineering foundation (code execution, web search, file/data tools) is now in place. Multi-agent systems will build on top of these — agents will use the existing tool registry, memory systems, and data tools as shared infrastructure.

---

*Document generated 2026-05-06. Last updated 2026-05-07 with all curriculum content through Notebook 16 (File and Data Tools).*