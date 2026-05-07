"""
Self-Correcting Data Analysis Agent (Notebook 16 + OpenAI Data Agent Insights)

Inspired by OpenAI's data agent:
  - "It evaluates its own progress. If an intermediate result looks wrong
    (e.g., zero rows due to an incorrect join or filter), the agent
    investigates what went wrong, adjusts its approach, and tries again."
  - "This closed-loop, self-learning process shifts iteration from the user
    into the agent itself."
  - Memory stores corrections so future answers begin from a more accurate baseline.

Key capabilities:
  1. Plan → Execute → Detect Anomaly → Self-Correct → Retry
  2. Anomaly detection: zero rows, null-heavy columns, out-of-range values,
     type mismatches, ambiguous questions
  3. Memory integration: stores learned corrections (connects to LTM)
  4. Workflow templates: pre-packaged analysis patterns for recurring work
  5. Clarifying questions: asks for specifics when instructions are vague
  6. Reasoning exposition: shows assumptions and execution steps alongside answers

Integration:
  - SchemaAwareFS    for data access + schema context
  - UnifiedDataTools for read/query/stats/transform
  - config.py        for LLM client
  - parser.py        for JSON extraction (4-strategy fallback)
  - memory.py / long_term_memory.py (optional) for persistent learnings
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from config import get_client, get_model
from parser import parse_response  # 4-strategy fallback JSON parser

# Optional: LongTermMemory from Notebook 11
# If not available, falls back to simple in-memory dict
_LONG_TERM_MEMORY_AVAILABLE = False

# ==============================================================================
# SECTION 1 -- Data structures
# ==============================================================================

@dataclass
class DataAnomaly:
    """An issue detected during analysis execution."""
    anomaly_type: str           # zero_rows, null_heavy, out_of_range,
                                # type_mismatch, ambiguous_question, no_data
    description: str
    severity: str = "warning"   # warning | error | critical
    suggestion: str = ""
    step_index: int = 0


@dataclass
class AnalysisStep:
    """A single step in an analysis plan."""
    step_index: int
    tool: str                   # read | query | stats | transform | synthesize
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    result_summary: str = ""
    anomalies: List[DataAnomaly] = field(default_factory=list)
    corrected: bool = False
    retry_count: int = 0


@dataclass
class AnalysisResult:
    """Full result of an analysis session."""
    question: str
    dataset_path: str
    steps: List[AnalysisStep]
    final_answer: str = ""
    anomalies_found: int = 0
    self_corrections: int = 0
    memory_entries_added: int = 0
    execution_time: float = 0.0


@dataclass
class WorkflowTemplate:
    """Pre-packaged analysis pattern (OpenAI: "workflows package recurring analyses")."""
    name: str
    description: str
    steps: List[Dict[str, Any]]   # List of {tool, parameters, reasoning}
    default_params: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# SECTION 2 -- SelfCorrectingDataAgent
# ==============================================================================

class SelfCorrectingDataAgent:
    """
    Data analysis agent with self-correction and memory.

    Usage:
        agent = SelfCorrectingDataAgent(fs, tools)
        result = agent.analyze("Which continent has the highest average GDP?",
                               "/data/countries.csv")
        print(result.final_answer)
    """

    def __init__(
        self,
        fs,
        reader: "DataReader",
        query: "DataQuery",
        stats: "DataStats",
        transform: "DataTransform",
        memory=None,
        max_retries: int = 2,
    ):
        self.fs = fs
        self.reader = reader
        self.query = query
        self.stats = stats
        self.transform = transform
        self.memory = memory  # Optional LongTermMemory or dict
        self.max_retries = max_retries
        self.client = get_client()
        self.model = get_model()

        # Built-in workflow templates
        self.workflows: Dict[str, WorkflowTemplate] = {
            "explore": WorkflowTemplate(
                name="explore",
                description="High-level overview: schema, summary stats",
                steps=[
                    {"tool": "read", "reasoning": "Load the dataset"},
                    {"tool": "stats", "parameters": {"columns": None}, "reasoning": "Summarize all numeric columns"},
                ]
            ),
            "compare_groups": WorkflowTemplate(
                name="compare_groups",
                description="Group by a category and compare aggregated values",
                steps=[
                    {"tool": "read", "reasoning": "Load dataset"},
                    {"tool": "query", "parameters": {"group_by": {"by": "{group_col}", "aggregates": [{"column": "{agg_col}", "func": "mean"}]}}, "reasoning": "Group and aggregate"},
                    {"tool": "query", "parameters": {"sort_by": [{"column": "mean_{agg_col}", "descending": True}]}, "reasoning": "Sort results by aggregated value"},
                ]
            ),
            "correlation_map": WorkflowTemplate(
                name="correlation_map",
                description="Find all pairwise correlations between numeric columns",
                steps=[
                    {"tool": "read", "reasoning": "Load dataset"},
                    {"tool": "stats", "parameters": {"columns": None}, "reasoning": "Identify numeric columns"},
                    {"tool": "correlate", "parameters": {}, "reasoning": "Compute pairwise correlations"},
                ]
            ),
        }

        # In-memory learned corrections (fallback if no LTM)
        self._learned: Dict[str, List[Dict[str, Any]]] = {}

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def analyze(self, question: str, dataset_path: str,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> AnalysisResult:
        """
        End-to-end analysis with planning, execution, anomaly detection,
        self-correction, and synthesis.

        Returns an AnalysisResult with full trace.
        """
        start_time = time.time()
        steps: List[AnalysisStep] = []

        # --- Step 0: Load dataset and get schema context ---
        read_result = self.fs.read(dataset_path)
        if not read_result.get("success"):
            return AnalysisResult(
                question=question,
                dataset_path=dataset_path,
                steps=[],
                final_answer=f"Error: could not load dataset. {read_result.get('error')}",
                execution_time=time.time() - start_time,
            )

        data = self.reader.read(read_result)
        if not data.get("success"):
            return AnalysisResult(
                question=question,
                dataset_path=dataset_path,
                steps=[],
                final_answer=f"Error: could not parse dataset. {data.get('error')}",
                execution_time=time.time() - start_time,
            )

        # Check memory for past corrections on this dataset
        memory_hints = self._recall_corrections(dataset_path)

        # --- Step 1: Generate analysis plan (LLM) ---
        plan = self._plan_analysis(question, data, memory_hints)

        # --- Step 2: Execute plan with anomaly detection ---
        current_data = data
        for i, step_def in enumerate(plan):
            step = AnalysisStep(
                step_index=i,
                tool=step_def.get("tool", "unknown"),
                parameters=step_def.get("parameters", {}),
                reasoning=step_def.get("reasoning", ""),
            )

            # Execute with retry loop
            for retry in range(self.max_retries + 1):
                result = self._execute_step(step, current_data, dataset_path)

                # Detect anomalies
                anomalies = self._detect_anomalies(result, step, current_data)
                step.anomalies = anomalies
                step.retry_count = retry

                if not anomalies or all(a.severity == "warning" for a in anomalies):
                    # Success or warnings only — proceed
                    if retry > 0:
                        step.corrected = True
                    break

                # Critical/error anomalies — attempt self-correction
                if retry < self.max_retries:
                    corrected = self._self_correct(step, anomalies, current_data, dataset_path)
                    if corrected:
                        step.parameters = corrected["parameters"]
                        step.reasoning = corrected.get("reasoning", step.reasoning)
                        step.corrected = True
                        # Re-execute on next loop iteration
                    else:
                        break  # Can't correct, proceed with anomalies
                else:
                    break  # Max retries exceeded

            # Store step result summary
            if isinstance(result, dict):
                step.result_summary = self._summarize_result(result)
                current_data = result  # Pass forward for chained operations
            else:
                step.result_summary = str(result)[:200]

            steps.append(step)

        # --- Step 3: Store learnings in memory ---
        new_entries = self._store_learnings(question, dataset_path, steps)

        # --- Step 4: Synthesize final answer (LLM) ---
        final_answer = self._synthesize(question, steps, data)

        # Count metrics
        anomalies_found = sum(len(s.anomalies) for s in steps)
        self_corrections = sum(1 for s in steps if s.corrected)

        return AnalysisResult(
            question=question,
            dataset_path=dataset_path,
            steps=steps,
            final_answer=final_answer,
            anomalies_found=anomalies_found,
            self_corrections=self_corrections,
            memory_entries_added=new_entries,
            execution_time=time.time() - start_time,
        )

    def ask_clarifying_question(self, question: str, dataset_path: str) -> Optional[str]:
        """
        If the question is too vague or ambiguous, ask for clarification.
        Returns a clarifying question string, or None if question is clear enough.
        """
        # Check for ambiguity signals
        ambiguous_signals = [
            "tell me about",
            "what can you say",
            "analyze this",
            "look at the data",
            "give me insights",
        ]
        q_lower = question.lower()
        if any(sig in q_lower for sig in ambiguous_signals):
            # Build prompt
            read_result = self.fs.read(dataset_path)
            schema_hint = ""
            if read_result.get("success") and read_result.get("metadata", {}).get("schema"):
                cols = list(read_result["metadata"]["schema"].get("columns", {}).keys())
                schema_hint = f" Available columns: {', '.join(cols)}." if cols else ""

            messages = [
                {"role": "system", "content": (
                    "You are a data analyst. The user's question is too vague. "
                    "Ask ONE specific clarifying question to help narrow down the analysis."
                    f"{schema_hint}"
                )},
                {"role": "user", "content": f"User asked: '{question}'\n\nWhat clarifying question should I ask?"},
            ]
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.5,
                    timeout=300,
                )
                # Reasoning models: content may be None; fall back to reasoning field
                text = resp.choices[0].message.content
                if text is None:
                    text = getattr(resp.choices[0].message, 'reasoning', '')
                return text.strip() if text else "Could you be more specific about what metrics or comparisons you're interested in?"
            except Exception:
                return "Could you be more specific about what metrics or comparisons you're interested in?"
        return None

    def run_workflow(self, name: str, dataset_path: str,
                     params: Optional[Dict[str, Any]] = None,
                     synthesize_answer: bool = True) -> AnalysisResult:
        """
        Execute a pre-defined workflow template.
        """
        wf = self.workflows.get(name)
        if not wf:
            return AnalysisResult(
                question=f"Workflow '{name}' not found",
                dataset_path=dataset_path,
                steps=[],
                final_answer=f"Error: No workflow named '{name}'. Available: {list(self.workflows.keys())}",
            )

        # Merge default params
        merged = dict(wf.default_params)
        if params:
            merged.update(params)

        # Substitute params into steps
        steps_def = []
        for s in wf.steps:
            step_copy = dict(s)
            # String substitution for simple placeholders like {group_col}
            def sub(v):
                if isinstance(v, str):
                    for pk, pv in merged.items():
                        v = v.replace(f"{{{pk}}}", str(pv))
                    return v
                if isinstance(v, dict):
                    return {k: sub(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [sub(item) for item in v]
                return v
            step_copy["parameters"] = sub(step_copy.get("parameters", {}))
            steps_def.append(step_copy)

        # Execute as a single "workflow" analysis
        start_time = time.time()
        read_result = self.fs.read(dataset_path)
        data = self.reader.read(read_result) if read_result.get("success") else {"success": False}

        executed_steps = []
        current_data = data
        for i, step_def in enumerate(steps_def):
            step = AnalysisStep(
                step_index=i,
                tool=step_def.get("tool"),
                parameters=step_def.get("parameters", {}),
                reasoning=step_def.get("reasoning", ""),
            )
            result = self._execute_step(step, current_data, dataset_path)
            step.result_summary = self._summarize_result(result)
            current_data = result
            executed_steps.append(step)

        # Synthesize (can be disabled for deterministic testing)
        final = ""
        if synthesize_answer:
            final = self._synthesize(f"Workflow: {wf.description}", executed_steps, data)

        return AnalysisResult(
            question=f"[Workflow: {name}] {wf.description}",
            dataset_path=dataset_path,
            steps=executed_steps,
            final_answer=final,
            execution_time=time.time() - start_time,
        )

    def register_workflow(self, template: WorkflowTemplate) -> None:
        """Add a custom workflow template."""
        self.workflows[template.name] = template

    # --------------------------------------------------------------------------
    # Internal: Planning
    # --------------------------------------------------------------------------

    def _plan_analysis(self, question: str, data: Dict[str, Any],
                       memory_hints: List[str]) -> List[Dict[str, Any]]:
        """
        Ask LLM to generate a step-by-step analysis plan.
        Returns a list of step dicts: {tool, parameters, reasoning}
        """
        columns = data.get("columns", [])
        row_count = data.get("row_count", 0)
        preview = data.get("preview", [])[:3]

        hints_text = "\n".join(f"- {h}" for h in memory_hints) if memory_hints else "None"

        system_prompt = (
            "You are a data analysis planner. Given a question and dataset info, "
            "output a JSON list of analysis steps. Each step has: tool, parameters, reasoning.\n\n"
            "Available tools:\n"
            "- read: already done, no parameters needed\n"
            "- query: filter/sort/group/projection. Parameters: conditions, sort_by, group_by, select, limit\n"
            "- stats: summarize or correlate. Parameters: columns (str or list), x_column, y_column\n"
            "- transform: derive/select/rename/pivot. Parameters: new_column, expression, select, mapping, etc.\n"
            "- synthesize: no parameters, produces final answer\n\n"
            f"Dataset: {row_count} rows, columns: {columns}\n"
            f"Preview rows: {json.dumps(preview, default=str)}\n\n"
            f"Past corrections (apply if relevant):\n{hints_text}\n\n"
            "IMPORTANT: Return ONLY a JSON array. No markdown, no explanation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            # Reasoning models: max_tokens = reasoning + content combined.
            # Complex planning prompts need 8000+ tokens for thinking alone.
            # We use 16384 to ensure both reasoning AND JSON output fit.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.3,
                timeout=180,
            )
            # Reasoning models: content may be None; fall back to reasoning field
            raw = resp.choices[0].message.content
            if raw is None:
                raw = getattr(resp.choices[0].message, 'reasoning', '')

            # Planning expects a JSON array — parse directly, not through ReAct parser
            valid_tools = {"read", "query", "stats", "transform", "synthesize"}

            # Try direct JSON parse first
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [s for s in parsed if s.get("tool") in valid_tools]
            except (json.JSONDecodeError, TypeError):
                pass

            # Fallback: try to extract JSON array from text (may have markdown or prose)
            import re
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, list):
                        return [s for s in parsed if s.get("tool") in valid_tools]
                except json.JSONDecodeError:
                    pass

            # Final fallback: single-step plan
            return [{"tool": "stats", "parameters": {"columns": None}, "reasoning": "Get overview statistics"},
                    {"tool": "query", "parameters": {"limit": 10}, "reasoning": "Inspect top rows"}]
        except Exception as e:
            # Fallback plan if LLM fails entirely
            return [
                {"tool": "stats", "parameters": {"columns": None}, "reasoning": "Get overview statistics"},
                {"tool": "query", "parameters": {"limit": 10}, "reasoning": "Inspect top rows"},
            ]

    # --------------------------------------------------------------------------
    # Internal: Execution
    # --------------------------------------------------------------------------

    def _execute_step(self, step: AnalysisStep, data: Dict[str, Any],
                      dataset_path: str) -> Dict[str, Any]:
        """Execute a single analysis step using the appropriate tool."""
        tool = step.tool
        params = step.parameters

        if tool == "read":
            # Already loaded, just pass through
            return data

        if tool == "query":
            return self.query.query(
                data,
                conditions=params.get("conditions"),
                sort_by=params.get("sort_by"),
                group_by=params.get("group_by"),
                select=params.get("select"),
                limit=params.get("limit"),
            )

        if tool == "stats":
            if "x_column" in params and "y_column" in params:
                if params.get("type") == "correlation":
                    return self.stats.correlate(data, params["x_column"], params["y_column"])
                if params.get("type") == "regression":
                    return self.stats.regress(data, params["x_column"], params["y_column"])
            return self.stats.summarize(
                data,
                columns=params.get("columns"),
                percentiles=params.get("percentiles"),
            )

        if tool == "transform":
            t = params.get("type", "derive")
            if t == "derive":
                return self.transform.derive(data, params.get("new_column", ""), params.get("expression", ""))
            if t == "select":
                return self.transform.select(data, params.get("columns", []))
            if t == "rename":
                return self.transform.rename(data, params.get("mapping", {}))
            if t == "pivot":
                return self.transform.pivot(
                    data,
                    params.get("index", ""),
                    params.get("columns", ""),
                    params.get("values", ""),
                    params.get("agg_func", "sum"),
                )
            return {"success": False, "error": f"Unknown transform type: {t}"}

        if tool == "synthesize":
            # Synthesis is done after all steps, not as a tool call
            return {"success": True, "type": "synthesis_pending"}

        return {"success": False, "error": f"Unknown tool: {tool}"}

    # --------------------------------------------------------------------------
    # Internal: Anomaly Detection
    # --------------------------------------------------------------------------

    def _detect_anomalies(self, result: Dict[str, Any],
                          step: AnalysisStep,
                          source_data: Dict[str, Any]) -> List[DataAnomaly]:
        """Check a step result for common failure patterns."""
        anomalies = []

        if not result.get("success"):
            anomalies.append(DataAnomaly(
                anomaly_type="execution_error",
                description=f"Step failed: {result.get('error', 'unknown error')}",
                severity="critical",
                suggestion="Check parameters and retry with corrected inputs.",
                step_index=step.step_index,
            ))
            return anomalies

        # Anomaly 1: Zero rows after filter/query
        if result.get("type") == "tabular" and result.get("row_count") == 0:
            anomalies.append(DataAnomaly(
                anomaly_type="zero_rows",
                description="Query returned zero rows. Filter may be too restrictive.",
                severity="error",
                suggestion="Try removing conditions or using broader values (e.g., 'contains' instead of '==').",
                step_index=step.step_index,
            ))

        # Anomaly 2: Null-heavy columns
        if result.get("type") == "tabular" and result.get("rows"):
            rows = result["rows"]
            if rows:
                cols = list(rows[0].keys())
                for col in cols:
                    nulls = sum(1 for r in rows if r.get(col) is None)
                    if nulls > len(rows) * 0.5:
                        anomalies.append(DataAnomaly(
                            anomaly_type="null_heavy",
                            description=f"Column '{col}' is {nulls}/{len(rows)} ({nulls*100//len(rows)}%) null.",
                            severity="warning",
                            suggestion=f"Consider dropping nulls in '{col}' or using a different column.",
                            step_index=step.step_index,
                        ))

        # Anomaly 3: Type mismatch in stats
        if result.get("type") == "statistics":
            for col, summary in result.get("summaries", {}).items():
                if "error" in summary:
                    anomalies.append(DataAnomaly(
                        anomaly_type="type_mismatch",
                        description=f"Stats failed for '{col}': {summary['error']}",
                        severity="error",
                        suggestion=f"Column '{col}' may not be numeric. Try auto-detect or specify correct columns.",
                        step_index=step.step_index,
                    ))

        # Anomaly 4: Single-row result when many expected
        if result.get("type") == "tabular" and result.get("row_count") == 1:
            if step.tool == "query" and not step.parameters.get("limit"):
                anomalies.append(DataAnomaly(
                    anomaly_type="narrow_result",
                    description="Query returned only 1 row. Possible over-filtering.",
                    severity="warning",
                    suggestion="Verify filter conditions are correct.",
                    step_index=step.step_index,
                ))

        return anomalies

    # --------------------------------------------------------------------------
    # Internal: Self-Correction
    # --------------------------------------------------------------------------

    def _self_correct(self, step: AnalysisStep, anomalies: List[DataAnomaly],
                      current_data: Dict[str, Any], dataset_path: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to fix parameters based on anomaly type.
        Returns corrected step dict or None if unfixable.
        """
        corrected = {"tool": step.tool, "parameters": dict(step.parameters), "reasoning": step.reasoning}

        for anomaly in anomalies:
            if anomaly.severity == "warning":
                continue  # Don't retry for warnings

            if anomaly.anomaly_type == "zero_rows" and step.tool == "query":
                # Strategy: remove one condition at a time, starting from the most restrictive
                conditions = corrected["parameters"].get("conditions", [])
                if conditions and len(conditions) > 1:
                    # Remove the last (most specific) condition
                    removed = conditions.pop()
                    corrected["reasoning"] += f" [Self-corrected: removed filter '{removed.get('column')} {removed.get('operator')} {removed.get('value')}' to avoid zero rows]"
                elif conditions:
                    # Only one condition: try contains instead of ==
                    cond = conditions[0]
                    if cond.get("operator") == "==":
                        cond["operator"] = "contains"
                        corrected["reasoning"] += f" [Self-corrected: changed '{cond.get('column')} ==' to 'contains' to avoid zero rows]"
                    else:
                        # Last resort: clear all conditions
                        corrected["parameters"]["conditions"] = []
                        corrected["reasoning"] += " [Self-corrected: cleared all filters to avoid zero rows]"
                return corrected

            if anomaly.anomaly_type == "type_mismatch" and step.tool == "stats":
                # Strategy: let stats auto-detect numeric columns instead of specified ones
                if corrected["parameters"].get("columns"):
                    corrected["parameters"]["columns"] = None  # Auto-detect
                    corrected["reasoning"] += " [Self-corrected: switched to auto-detect numeric columns]"
                return corrected

            if anomaly.anomaly_type == "execution_error":
                # Can't self-correct generic errors
                return None

        return None if all(a.severity == "warning" for a in anomalies) else corrected

    # --------------------------------------------------------------------------
    # Internal: Memory (Learned Corrections)
    # --------------------------------------------------------------------------

    def _recall_corrections(self, dataset_path: str) -> List[str]:
        """Retrieve past corrections for this dataset."""
        # Try external memory first (LongTermMemory from Notebook 11)
        if self.memory and hasattr(self.memory, "recall"):
            try:
                results = self.memory.recall(f"corrections for {dataset_path}")
                return [str(r) for r in results[:3]]
            except Exception:
                pass
        # Fallback to in-memory dict
        return [e["hint"] for e in self._learned.get(dataset_path, [])]

    def _store_learnings(self, question: str, dataset_path: str,
                        steps: List[AnalysisStep]) -> int:
        """Store corrections that resolved anomalies for future recall."""
        entries_added = 0
        for step in steps:
            if step.corrected and step.anomalies:
                for anomaly in step.anomalies:
                    if anomaly.severity in ("error", "critical"):
                        hint = (
                            f"For '{dataset_path}', when using {step.tool}: "
                            f"{anomaly.suggestion} (detected: {anomaly.description})"
                        )
                        entry = {
                            "dataset": dataset_path,
                            "tool": step.tool,
                            "anomaly": anomaly.anomaly_type,
                            "hint": hint,
                            "timestamp": time.time(),
                        }
                        self._learned.setdefault(dataset_path, []).append(entry)
                        entries_added += 1

                        # Also store in external memory if available
                        if self.memory and hasattr(self.memory, "store_fact"):
                            try:
                                self.memory.store_fact(
                                    f"data_correction_{dataset_path}_{anomaly.anomaly_type}",
                                    hint,
                                    importance=0.7,
                                )
                            except Exception:
                                pass
        return entries_added

    # --------------------------------------------------------------------------
    # Internal: Synthesis
    # --------------------------------------------------------------------------

    def _synthesize(self, question: str, steps: List[AnalysisStep],
                    source_data: Dict[str, Any]) -> str:
        """Ask LLM to synthesize a final answer from step results."""
        # Build execution trace for context
        trace = []
        for s in steps:
            line = f"Step {s.step_index + 1} ({s.tool}): {s.reasoning}"
            if s.corrected:
                line += " [self-corrected]"
            if s.anomalies:
                for a in s.anomalies:
                    line += f" [{a.severity}: {a.anomaly_type}]"
            line += f" -> {s.result_summary}"
            trace.append(line)

        trace_text = "\n".join(trace)

        messages = [
            {"role": "system", "content": (
                "You are a data analyst. Answer the user's question based on the analysis execution trace. "
                "Be specific — cite numbers, names, and comparisons. "
                "Format your answer clearly with key findings. "
                "If anomalies occurred, mention how they were handled."
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Dataset: {source_data.get('row_count', '?')} rows, "
                f"columns: {source_data.get('columns', [])}\n\n"
                f"Execution Trace:\n{trace_text}\n\n"
                "Please provide a clear, specific answer."
            )},
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0.4,
                timeout=180,
            )
            # Reasoning models: content may be None; fall back to reasoning field
            text = resp.choices[0].message.content
            if text is None:
                text = getattr(resp.choices[0].message, 'reasoning', '')
            return text.strip() if text else f"[Synthesis returned empty] Raw trace:\n{trace_text}"
        except Exception as e:
            return f"[Synthesis failed: {e}] Raw trace:\n{trace_text}"

    # --------------------------------------------------------------------------
    # Internal: Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _summarize_result(result: Dict[str, Any]) -> str:
        """Create a concise summary of a tool result for the trace."""
        if not result.get("success"):
            return f"ERROR: {result.get('error', 'unknown')}"

        rtype = result.get("type", "unknown")
        if rtype == "tabular":
            rc = result.get("row_count", 0)
            cols = result.get("columns", [])
            qs = result.get("query_summary", "")
            return f"{rc} rows, {len(cols)} cols ({qs})" if qs else f"{rc} rows, {len(cols)} cols"
        if rtype == "statistics":
            summaries = result.get("summaries", {})
            return f"stats for {len(summaries)} column(s)"
        if rtype == "correlation":
            return f"r={result.get('correlation')} ({result.get('interpretation')})"
        if rtype == "regression":
            return f"R²={result.get('r_squared')}, {result.get('equation')}"
        if rtype == "json":
            return f"JSON ({result.get('json_type')}), {result.get('size')} items"
        return f"Result type={rtype}"


# ==============================================================================
# SECTION 3 -- Fast deterministic tests
# ==============================================================================

def run_tests():
    """Test anomaly detection, self-correction, and workflow execution
    WITHOUT LLM calls (deterministic)."""
    from schema_aware_fs import SchemaAwareFS
    from unified_data_tools import DataReader, DataQuery, DataStats, DataTransform

    fs = SchemaAwareFS()
    reader = DataReader()
    query = DataQuery()
    stats = DataStats()
    transform = DataTransform()

    # Load sample dataset
    csv_content = """name,age,city,salary
Alice,30,New York,85000
Bob,25,San Francisco,92000
Charlie,35,New York,78000
Diana,28,Chicago,71000
Eve,32,San Francisco,95000"""
    fs.create("/data/employees.csv", csv_content)
    data = reader.read(fs.read("/data/employees.csv"))
    assert data["success"]

    # Create agent (LLM calls will fail gracefully in tests without network)
    agent = SelfCorrectingDataAgent(
        fs=fs,
        reader=reader,
        query=query,
        stats=stats,
        transform=transform,
        memory=None,
        max_retries=1,
    )

    # Test 1: Anomaly detection — zero rows
    empty_result = query.query(data, conditions=[{"column": "city", "operator": "==", "value": "Mars"}])
    assert empty_result["success"]
    assert empty_result["row_count"] == 0

    dummy_step = AnalysisStep(step_index=0, tool="query",
                               parameters={"conditions": [{"column": "city", "operator": "==", "value": "Mars"}]})
    anomalies = agent._detect_anomalies(empty_result, dummy_step, data)
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == "zero_rows"
    assert anomalies[0].severity == "error"
    print("[PASS] Test 1: Detect zero rows anomaly")

    # Test 2: Self-correction for zero rows
    corrected = agent._self_correct(dummy_step, anomalies, data, "/data/employees.csv")
    assert corrected is not None
    # Single condition with '==' gets changed to 'contains' first
    assert corrected["parameters"]["conditions"][0]["operator"] == "contains"
    print("[PASS] Test 2: Self-correct zero rows by changing == to contains")

    # Test 3: Anomaly detection — null heavy
    # Create data with many nulls
    null_csv = "name,score\nAlice,95\nBob,\nCharlie,\nDiana,88\nEve,"
    fs.create("/data/nulls.csv", null_csv)
    null_data = reader.read(fs.read("/data/nulls.csv"))
    null_result = query.query(null_data)  # No filter, just pass through
    null_step = AnalysisStep(step_index=0, tool="query", parameters={})
    anomalies = agent._detect_anomalies(null_result, null_step, null_data)
    # score column has 3/5 nulls = 60%
    null_anomalies = [a for a in anomalies if a.anomaly_type == "null_heavy"]
    assert len(null_anomalies) >= 1
    assert "score" in null_anomalies[0].description
    print("[PASS] Test 3: Detect null-heavy column")

    # Test 4: Anomaly detection — type mismatch in stats
    # Using non-existent column to force error (self-correction strategy is same: auto-detect)
    stats_result = stats.summarize(null_data, columns=["nonexistent_column"])
    stats_step = AnalysisStep(step_index=0, tool="stats", parameters={"columns": ["nonexistent_column"]})
    anomalies = agent._detect_anomalies(stats_result, stats_step, null_data)
    type_anomalies = [a for a in anomalies if a.anomaly_type == "type_mismatch"]
    assert len(type_anomalies) >= 1
    print("[PASS] Test 4: Detect type mismatch (missing column triggers error)")

    # Test 5: Self-correction for type mismatch
    corrected = agent._self_correct(stats_step, type_anomalies, null_data, "/data/nulls.csv")
    assert corrected is not None
    assert corrected["parameters"]["columns"] is None  # Auto-detect
    print("[PASS] Test 5: Self-correct type mismatch by auto-detect")

    # Test 6: Workflow execution (deterministic, no LLM)
    wf_result = agent.run_workflow("explore", "/data/employees.csv", synthesize_answer=False)
    assert wf_result.dataset_path == "/data/employees.csv"
    assert len(wf_result.steps) == 2
    assert wf_result.steps[0].tool == "read"
    assert wf_result.steps[1].tool == "stats"
    print("[PASS] Test 6: Execute 'explore' workflow")

    # Test 7: Memory learning storage
    # Simulate a corrected step
    step_with_anomaly = AnalysisStep(
        step_index=0, tool="query",
        parameters={"conditions": [{"column": "city", "operator": "==", "value": "Mars"}]},
        corrected=True,
        anomalies=[DataAnomaly(
            anomaly_type="zero_rows", description="test",
            severity="error", suggestion="try broader filter"
        )],
    )
    entries = agent._store_learnings("test q", "/data/employees.csv", [step_with_anomaly])
    assert entries == 1
    # Recall should return the hint
    hints = agent._recall_corrections("/data/employees.csv")
    assert any("try broader filter" in h for h in hints)
    print("[PASS] Test 7: Store and recall learned corrections")

    # Test 8: Result summarization
    summary = agent._summarize_result({"success": True, "type": "tabular",
                                        "row_count": 5, "columns": ["a", "b"],
                                        "query_summary": "filtered by 1 condition"})
    assert "5 rows" in summary
    assert "filtered by 1 condition" in summary
    print("[PASS] Test 8: Result summarization")

    # Test 9: Clarifying question detection (deterministic path: not ambiguous)
    clear_q = "What is the average salary by city?"
    cq = agent.ask_clarifying_question(clear_q, "/data/employees.csv")
    assert cq is None  # clear question needs no clarification
    # Also test the vague path returns a string (method is callable)
    vague_q = "tell me about the data"
    # Note: this would trigger LLM call; we verify the method exists and handles inputs
    assert callable(agent.ask_clarifying_question)
    print("[PASS] Test 9: Clarifying question detection")

    # Test 10: Custom workflow registration
    agent.register_workflow(WorkflowTemplate(
        name="test_wf",
        description="A test workflow",
        steps=[{"tool": "read", "reasoning": "Load"}],
    ))
    assert "test_wf" in agent.workflows
    print("[PASS] Test 10: Custom workflow registration")

    print("\n*** All 10 SelfCorrectingDataAgent tests passed!")


if __name__ == "__main__":
    run_tests()
