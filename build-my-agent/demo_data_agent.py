"""
Data Analysis Agent Demo -- Real LLM Integration (Notebook 16 Capstone)

This demo exercises the full self-correcting data agent with:
  1. SchemaAwareFS          -> load countries dataset with rich metadata
  2. UnifiedDataTools       -> query, stats, transform
  3. SelfCorrectingDataAgent -> plan with LLM, detect anomalies, self-correct, synthesize
  4. Workflow templates     -> pre-packaged analysis patterns
  5. Evaluation cases       -> golden dataset with expected answers

Inspired by OpenAI's in-house data agent lessons:
  - "Less is More"          -> unified tool interface (4 classes vs 6)
  - "Guide the Goal"        -> LLM plans high-level, tools execute deterministically
  - "Meaning Lives in Code" -> derivation chains track data lineage
  - "Context is Everything"   -> 6-layer metadata (schema, usage, annotations, lineage, memory, runtime)

Usage:
    cd build-my-agent && python demo_data_agent.py

If the LLM endpoint is unavailable, the demo falls back to deterministic
execution trace display.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import csv
import io
import urllib.request

from schema_aware_fs import SchemaAwareFS
from unified_data_tools import DataReader, DataQuery, DataStats, DataTransform
from self_correcting_data_agent import SelfCorrectingDataAgent, WorkflowTemplate


# ==============================================================================
# SECTION 1 -- Countries Dataset (from Notebook 16, expanded for richer analysis)
# ==============================================================================

COUNTRIES_CSV = """name,population_millions,area_km2,gdp_billions_usd,continent
China,1412,9597000,17963,Asia
India,1408,3287000,3385,Asia
United States,334,9834000,25462,North America
Indonesia,275,1905000,1319,Asia
Pakistan,229,881000,377,Asia
Brazil,214,8516000,1920,South America
Nigeria,218,924000,477,Africa
Bangladesh,170,148000,460,Asia
Russia,144,17098000,2240,Europe
Mexico,128,1964000,1414,North America
Japan,125,378000,4231,Asia
Ethiopia,120,1104000,126,Africa
Egypt,104,1002000,477,Africa
Germany,83,357000,4259,Europe
France,68,641000,2783,Europe
United Kingdom,67,243000,3071,Europe
Italy,59,301000,2010,Europe
South Korea,52,100000,1665,Asia
Australia,26,7692000,1675,Oceania
Canada,39,9985000,2140,North America"""


# ==============================================================================
# SECTION 2 -- Evaluation Cases (golden dataset for correctness checking)
# ==============================================================================

@dataclass
class EvalCase:
    name: str
    question: str
    expected_answer_type: str          # scalar | table | comparison | ranking
    expected_contains: List[str]       # substrings expected in correct answer
    tolerance: Optional[float] = None # for numeric answers

EVAL_CASES = [
    EvalCase(
        name="asia_gdp_total",
        question="What is the total GDP of all Asian countries?",
        expected_answer_type="scalar",
        expected_contains=["29", "400"],  # ~$29,400 billion (7 Asian countries)
    ),
    EvalCase(
        name="top_gdp_per_capita",
        question="Which country has the highest GDP per capita?",
        expected_answer_type="ranking",
        expected_contains=["United States", "GDP per capita"],
    ),
    EvalCase(
        name="continent_comparison",
        question="Compare average GDP per country between Asia and Europe.",
        expected_answer_type="comparison",
        expected_contains=["Asia", "Europe", "average", "GDP"],
    ),
    EvalCase(
        name="population_density",
        question="What are the top 3 most densely populated countries?",
        expected_answer_type="ranking",
        expected_contains=["Bangladesh", "South Korea", "India", "density"],
    ),
    EvalCase(
        name="pop_gdp_correlation",
        question="Is there a correlation between population and GDP?",
        expected_answer_type="scalar",
        expected_contains=["correlation", "0."],
    ),
]


# ==============================================================================
# SECTION 3 -- Demo Runner
# ==============================================================================

def run_demo():
    """Execute the full demo with real LLM integration."""
    print("=" * 72)
    print("  DATA ANALYSIS AGENT DEMO -- Notebook 16 + OpenAI Insights")
    print("=" * 72)

    # --- Setup filesystem and tools ---
    fs = SchemaAwareFS()
    reader = DataReader()
    query = DataQuery()
    stats = DataStats()
    transform = DataTransform()

    # --- Load countries dataset ---
    print("\n[1] Loading countries dataset into SchemaAwareFS...")
    result = fs.create("/data/countries.csv", COUNTRIES_CSV)
    assert result["success"]
    meta = fs.metadata["/data/countries.csv"]
    print(f"    Created: {result['path']} ({result['size_bytes']} bytes)")
    print(f"    Content type: {result['content_type']}")
    print(f"    Schema inferred: {result['schema_inferred']}")
    print(f"    Columns: {list(meta.schema['columns'].keys())}")
    for col, info in meta.schema["columns"].items():
        print(f"      - {col}: {info['type']} (nulls={info['null_count']}, unique={info['unique_count']})")

    # --- Add human annotations (Layer 2 context) ---
    fs.metadata["/data/countries.csv"].annotate(
        "description",
        "20 most populous countries with population (millions), area (km2), GDP (billions USD), and continent"
    )
    fs.metadata["/data/countries.csv"].annotate(
        "caveats",
        "GDP figures are nominal (not PPP). Population is 2023 estimates."
    )
    print("\n[2] Added human annotations (Layer 2 context)")

    # --- Read and parse data ---
    data = reader.read(fs.read("/data/countries.csv"))
    assert data["success"]
    print(f"\n[3] Parsed dataset: {data['row_count']} rows, {len(data['columns'])} columns")
    print(f"    First 3 rows:")
    for r in data["rows"][:3]:
        print(f"      {r['name']:20s} pop={r['population_millions']:>6.0f}M  "
              f"area={r['area_km2']:>10,.0f}km2  GDP=${r['gdp_billions_usd']:>7,.0f}B  {r['continent']}")

    # --- Demonstrate unified query ---
    print("\n[4] Unified query demo: Asian countries, sorted by GDP descending, top 5")
    asia = query.query(data,
                       conditions=[{"column": "continent", "operator": "==", "value": "Asia"}],
                       sort_by=[{"column": "gdp_billions_usd", "descending": True}],
                       select=["name", "population_millions", "gdp_billions_usd"],
                       limit=5)
    assert asia["success"]
    print(f"    Result: {asia['row_count']} rows")
    for r in asia["rows"]:
        print(f"      {r['name']:20s} ${r['gdp_billions_usd']:>8,.0f}B")

    # --- Demonstrate statistics ---
    print("\n[5] Statistics demo: GDP summary")
    gdp_stats = stats.summarize(data, columns=["gdp_billions_usd"])
    assert gdp_stats["success"]
    s = gdp_stats["summaries"]["gdp_billions_usd"]
    print(f"    count={s['count']}  mean=${s['mean']:,.0f}B  median=${s['median']:,.0f}B")
    print(f"    std=${s['std_dev']:,.0f}B  min=${s['min']:,.0f}B  max=${s['max']:,.0f}B")
    print(f"    IQR=${s['iqr']:,.0f}B  (p25=${s['p25']:,.0f}B, p75=${s['p75']:,.0f}B)")

    # --- Demonstrate correlation ---
    print("\n[6] Correlation: population vs GDP")
    corr = stats.correlate(data, "population_millions", "gdp_billions_usd")
    assert corr["success"]
    print(f"    Pearson r = {corr['correlation']} ({corr['interpretation']})")

    # --- Demonstrate transform: derive GDP per capita ---
    print("\n[7] Transform: derive GDP per capita (thousands USD)")
    with_gdpc = transform.derive(data, "gdp_per_capita_k",
                                  lambda r: round(float(r["gdp_billions_usd"]) * 1000 /
                                                  float(r["population_millions"]), 1))
    assert with_gdpc["success"]
    # Sort and show top 5
    ranked = query.query(with_gdpc, sort_by=[{"column": "gdp_per_capita_k", "descending": True}],
                          select=["name", "continent", "gdp_per_capita_k"], limit=5)
    print(f"    Top 5 by GDP per capita:")
    for r in ranked["rows"]:
        print(f"      {r['name']:20s} {r['continent']:15s} ${r['gdp_per_capita_k']:>8,.1f}k")

    # --- Save derived file with lineage ---
    derived_csv = _rows_to_csv(with_gdpc["rows"], with_gdpc["columns"])
    fs.derive("/data/countries.csv", "/data/countries_with_gdpc.csv",
              "derive_gdp_per_capita", derived_csv,
              {"expression": "gdp * 1000 / population"})
    lineage = fs.get_lineage("/data/countries_with_gdpc.csv")
    print(f"\n[8] Saved derived file with lineage:")
    for entry in lineage["derivation_chain"]:
        print(f"    {entry['operation']} from {entry.get('source', 'unknown')}")

    # --- Demonstrate anomaly detection (deterministic) ---
    print("\n[9] Anomaly detection demo: query with impossible filter")
    empty = query.query(data, conditions=[{"column": "continent", "operator": "==", "value": "Antarctica"}])
    print(f"    Filter 'continent == Antarctica' -> {empty['row_count']} rows")
    print(f"    Anomaly: zero_rows (this would trigger self-correction in the agent)")

    # --- Create the self-correcting agent ---
    print("\n[10] Initializing SelfCorrectingDataAgent...")
    agent = SelfCorrectingDataAgent(
        fs=fs,
        reader=reader,
        query=query,
        stats=stats,
        transform=transform,
        memory=None,
        max_retries=1,
    )
    print("    Agent ready. Tools: read, query, stats, transform, synthesize")
    print("    Workflows:", list(agent.workflows.keys()))

    # --- Run workflow (deterministic, no LLM) ---
    print("\n[11] Running 'explore' workflow on countries dataset...")
    wf_result = agent.run_workflow("explore", "/data/countries.csv", synthesize_answer=False)
    print(f"    Workflow completed in {wf_result.execution_time:.2f}s")
    print(f"    Steps executed: {len(wf_result.steps)}")
    for s in wf_result.steps:
        print(f"      Step {s.step_index + 1}: {s.tool} -> {s.result_summary}")

    # --- Register custom workflow ---
    print("\n[12] Registering custom 'gdp_per_capita_ranking' workflow...")
    agent.register_workflow(WorkflowTemplate(
        name="gdp_per_capita_ranking",
        description="Rank all countries by GDP per capita",
        steps=[
            {"tool": "read", "reasoning": "Load dataset"},
            {"tool": "transform", "parameters": {"type": "derive", "new_column": "gdp_per_capita_k",
              "expression": "{gdp_billions_usd} * 1000 / {population_millions}"}, "reasoning": "Derive GDP per capita"},
            {"tool": "query", "parameters": {"sort_by": [{"column": "gdp_per_capita_k", "descending": True}],
              "select": ["name", "continent", "gdp_per_capita_k"]}, "reasoning": "Sort descending by GDP per capita"},
        ],
    ))
    wf2 = agent.run_workflow("gdp_per_capita_ranking", "/data/countries.csv", synthesize_answer=False)
    print(f"    Custom workflow completed: {len(wf2.steps)} steps")
    print(f"    Top country: {wf2.steps[-1].result_summary}")

    # --- Evaluation: deterministic correctness checks ---
    print("\n[13] Running evaluation cases (deterministic correctness)...")
    eval_pass = 0
    eval_fail = 0

    for case in EVAL_CASES:
        # Execute the question using deterministic tool chain (no LLM for eval)
        # We build the answer from tool results directly
        answer = _deterministic_answer(case, data, query, stats, transform)

        # Check expected_contains
        found = [s for s in case.expected_contains if s.lower() in answer.lower()]
        if len(found) >= max(1, len(case.expected_contains) // 2):
            status = "PASS"
            eval_pass += 1
        else:
            status = "FAIL"
            eval_fail += 1

        print(f"    [{status}] {case.name}: {case.question[:50]}...")
        print(f"           Expected: {case.expected_contains}")
        print(f"           Found:    {found}")
        print(f"           Answer:   {answer[:100]}...")

    print(f"\n    Evaluation: {eval_pass}/{eval_pass + eval_fail} cases passed")

    # --- LLM connectivity check ---
    llm_available = False
    try:
        # Quick connectivity probe to the LLM base URL
        from config import LLM_CONFIG
        base = LLM_CONFIG["base_url"].replace("/v1", "")
        req = urllib.request.Request(base, method="HEAD")
        urllib.request.urlopen(req, timeout=3)
        llm_available = True
    except Exception:
        pass

    if not llm_available:
        print("\n" + "=" * 72)
        print("  LLM ENDPOINT UNREACHABLE -- showing deterministic traces only")
        print("=" * 72)
    else:
        print("\n" + "=" * 72)
        print("  LLM-POWERED ANALYSIS (endpoint reachable)")
        print("=" * 72)

    llm_questions = [
        "Which continent has the highest total GDP and which has the highest average GDP per country?",
        "What is the relationship between population and GDP? Are larger countries richer?",
        "Which are the top 5 countries by GDP per capita, and which continent dominates?",
    ]

    for q in llm_questions:
        print(f"\n[LLM] Question: {q}")
        if llm_available:
            try:
                start = time.time()
                result = agent.analyze(q, "/data/countries.csv")
                elapsed = time.time() - start

                print(f"    Analysis completed in {elapsed:.1f}s")
                print(f"    Steps: {len(result.steps)} | Anomalies: {result.anomalies_found} | "
                      f"Self-corrections: {result.self_corrections}")
                print(f"    Final Answer:\n{'-' * 60}")
                for line in result.final_answer.split("\n")[:10]:
                    print(f"      {line}")
                if len(result.final_answer.split("\n")) > 10:
                    print(f"      ... ({len(result.final_answer.split(chr(10))) - 10} more lines)")
                print(f"{'-' * 60}")

                # Write answer to filesystem
                fs.write(f"/answers/{_safe_filename(q[:40])}.txt", result.final_answer)

            except Exception as e:
                print(f"    [LLM error during analysis: {e}]")
                print(f"    (Falling back to deterministic trace)")
                _show_deterministic_trace(q, data, query, stats, transform)
        else:
            print(f"    [LLM endpoint unreachable -- deterministic trace only]")
            _show_deterministic_trace(q, data, query, stats, transform)

    # --- Final summary ---
    print("\n" + "=" * 72)
    print("  DEMO SUMMARY")
    print("=" * 72)
    print(f"  Filesystem:        {fs.stats()['total_files']} files, {fs.stats()['total_bytes']} bytes")
    print(f"  Evaluation cases:  {eval_pass}/{eval_pass + eval_fail} passed")
    print(f"  Workflows:         {len(agent.workflows)} registered")
    print(f"  Memory entries:    {len(agent._learned)}")
    print(f"  Files in /answers: {len(fs.list_files('/answers')['files'])}")
    print("=" * 72)


# ==============================================================================
# SECTION 4 -- Deterministic answer builder (for evaluation, no LLM)
# ==============================================================================

def _deterministic_answer(case: EvalCase, data, query, stats, transform) -> str:
    """Build an answer from deterministic tool execution (no LLM)."""

    if case.name == "asia_gdp_total":
        asia = query.query(data, conditions=[{"column": "continent", "operator": "==", "value": "Asia"}])
        total = sum(float(r["gdp_billions_usd"]) for r in asia["rows"] if r.get("gdp_billions_usd"))
        return f"Total GDP of Asian countries: ${total:,.0f} billion. " \
               f"Countries included: {len(asia['rows'])}."

    if case.name == "top_gdp_per_capita":
        d = transform.derive(data, "gdp_per_capita_k",
                             lambda r: float(r["gdp_billions_usd"]) * 1000 / float(r["population_millions"]))
        top = query.query(d, sort_by=[{"column": "gdp_per_capita_k", "descending": True}], limit=1)
        r = top["rows"][0]
        return f"{r['name']} has the highest GDP per capita at ${r['gdp_per_capita_k']:,.1f}k."

    if case.name == "continent_comparison":
        groups = query.query(data, group_by={
            "by": "continent",
            "aggregates": [{"column": "gdp_billions_usd", "func": "mean", "alias": "avg_gdp"}]
        })
        rows = sorted(groups["rows"], key=lambda x: x.get("avg_gdp", 0), reverse=True)
        top = rows[0]
        return f"{top['continent']} has the highest average GDP per country at ${top['avg_gdp']:,.0f}B."

    if case.name == "population_density":
        d = transform.derive(data, "density",
                             lambda r: float(r["population_millions"]) * 1e6 / float(r["area_km2"]))
        top3 = query.query(d, sort_by=[{"column": "density", "descending": True}], limit=3)
        names = ", ".join(r["name"] for r in top3["rows"])
        return f"Top 3 most densely populated: {names}."

    if case.name == "pop_gdp_correlation":
        c = stats.correlate(data, "population_millions", "gdp_billions_usd")
        return f"Correlation between population and GDP: r = {c['correlation']} ({c['interpretation']})."

    return "[Could not generate deterministic answer]"


def _show_deterministic_trace(question: str, data, query, stats, transform) -> None:
    """Show what the deterministic tools would compute for this question."""
    # Heuristic: check for keywords and run matching tools
    q_lower = question.lower()

    if "gdp per capita" in q_lower or "per capita" in q_lower:
        d = transform.derive(data, "gdp_per_capita_k",
                               lambda r: float(r["gdp_billions_usd"]) * 1000 / float(r["population_millions"]))
        top5 = query.query(d, sort_by=[{"column": "gdp_per_capita_k", "descending": True}],
                            select=["name", "continent", "gdp_per_capita_k"], limit=5)
        print(f"    [Deterministic] Top 5 by GDP per capita:")
        for r in top5["rows"]:
            print(f"      {r['name']:20s} {r['continent']:15s} ${r['gdp_per_capita_k']:>8,.1f}k")

    elif "correlation" in q_lower:
        c = stats.correlate(data, "population_millions", "gdp_billions_usd")
        print(f"    [Deterministic] Population-GDP correlation: r = {c['correlation']} ({c['interpretation']})")

    elif "continent" in q_lower and ("total" in q_lower or "average" in q_lower or "gdp" in q_lower):
        groups = query.query(data, group_by={
            "by": "continent",
            "aggregates": [
                {"column": "gdp_billions_usd", "func": "sum", "alias": "total_gdp"},
                {"column": "gdp_billions_usd", "func": "mean", "alias": "avg_gdp"},
            ]
        })
        rows = sorted(groups["rows"], key=lambda x: x.get("total_gdp", 0), reverse=True)
        print(f"    [Deterministic] Continent GDP ranking:")
        for r in rows:
            print(f"      {r['continent']:20s} total=${r.get('total_gdp', 0):>10,.0f}B  avg=${r.get('avg_gdp', 0):>8,.0f}B")

    else:
        s = stats.summarize(data)
        print(f"    [Deterministic] Dataset overview: {s['summaries']}")


def _rows_to_csv(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Convert rows back to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for r in rows:
        writer.writerow({c: str(r.get(c, "")) for c in columns})
    return output.getvalue()


def _safe_filename(text: str) -> str:
    """Create a filesystem-safe filename from text."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).lower()


# ==============================================================================
# SECTION 5 -- Main
# ==============================================================================

if __name__ == "__main__":
    run_demo()
