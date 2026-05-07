"""
Live LLM Test for Notebook 16 — SelfCorrectingDataAgent
========================================================
End-to-end test with REAL LLM calls: planning + synthesis.

Handles cold-start by polling the endpoint with exponential backoff
(up to 5 minutes), then runs agent.analyze() with 180-second timeout.

Usage:
    python live_test_data_agent.py
"""

import json
import time
import urllib.request
import urllib.error

from openai import OpenAI

from schema_aware_fs import SchemaAwareFS
from unified_data_tools import DataReader, DataQuery, DataStats, DataTransform
from self_correcting_data_agent import SelfCorrectingDataAgent
from config import LLM_CONFIG, get_model


# =============================================================================
# 1. Endpoint readiness check (handles cold-start)
# =============================================================================

def wait_for_endpoint(base_url: str, max_wait: int = 300, interval: int = 10) -> bool:
    """
    Poll the LLM endpoint until it responds to a GET /models request.
    Cold-start on Modal can take 60-120s (container boot + model load).
    Returns True if endpoint is reachable, False after max_wait.
    """
    models_url = base_url.rstrip("/") + "/models"

    print(f"  Checking endpoint: {base_url}")
    print(f"  Max wait: {max_wait}s (polling every {interval}s)...")

    attempts = 0
    start = time.time()
    while time.time() - start < max_wait:
        attempts += 1
        elapsed = time.time() - start
        try:
            req = urllib.request.Request(models_url, method="GET")
            req.add_header("Authorization", f"Bearer {LLM_CONFIG['api_key']}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print(f"  [READY] Endpoint available after {elapsed:.0f}s (attempt {attempts})")
                    return True
        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            print(f"  [WAIT] Attempt {attempts} ({elapsed:.0f}s): {reason}")
        except Exception as e:
            print(f"  [WAIT] Attempt {attempts} ({elapsed:.0f}s): {type(e).__name__}: {e}")

        time.sleep(interval)

    print(f"  [TIMEOUT] Endpoint not reachable after {max_wait}s")
    return False


# =============================================================================
# 2. Test dataset
# =============================================================================

COUNTRIES_CSV = """name,population_millions,area_km2,gdp_billions_usd,continent
China,1412,9597000,17963,Asia
India,1408,3287000,3385,Asia
United States,334,9834000,25462,North America
Indonesia,276,1905000,1319,Asia
Pakistan,231,881912,376,Asia
Brazil,215,8516000,1920,South America
Nigeria,219,923768,441,Africa
Bangladesh,171,148460,416,Asia
Russia,144,17098000,1776,Europe
Mexico,129,1964000,1322,North America
Japan,125,377975,4231,Asia
Ethiopia,123,1104300,111,Africa
Philippines,114,300000,404,Asia
Egypt,109,1002450,405,Africa
Germany,84,357022,4072,Europe
United Kingdom,68,242495,3070,Europe
France,68,640679,2780,Europe
Italy,59,301340,2010,Europe
South Korea,52,100210,1665,Asia
Australia,26,7692000,1675,Oceania"""


# =============================================================================
# 3. Live test runner
# =============================================================================

def run_live_test(question: str, timeout: int = 180):
    """
    Full end-to-end: setup filesystem, create agent, call analyze().
    Uses a client with the given timeout to accommodate cold-start.
    """
    print("\n" + "=" * 70)
    print(f"  LIVE LLM TEST -- SelfCorrectingDataAgent")
    print(f"  Question: \"{question}\"")
    print(f"  Request timeout: {timeout}s")
    print("=" * 70)

    # --- Phase 1: Wait for endpoint ---
    print("\n[Phase 1] Checking LLM endpoint readiness...")
    if not wait_for_endpoint(LLM_CONFIG["base_url"], max_wait=300, interval=15):
        print("\n[ABORT] LLM endpoint unreachable after 5 minutes.")
        print("The Modal server may be scaled to zero. Try again later.")
        return False

    # --- Phase 2: Create long-timeout client ---
    print(f"\n[Phase 2] Creating OpenAI client with timeout=600s...")
    live_client = OpenAI(
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
        timeout=600,
    )
    model_name = get_model()
    print(f"  Model: {model_name}")

    # --- Phase 3: Setup filesystem + tools ---
    print("\n[Phase 3] Setting up SchemaAwareFS + unified data tools...")
    fs = SchemaAwareFS()
    fs.create("/data/countries.csv", COUNTRIES_CSV)
    read_result = fs.read("/data/countries.csv")
    assert read_result["success"], f"FS read failed: {read_result}"

    reader = DataReader()
    query_tool = DataQuery()
    stats = DataStats()
    transform = DataTransform()

    # Parse to verify
    data = reader.read(read_result)
    print(f"  Loaded: {data['row_count']} rows, {data['columns']}")

    # --- Phase 4: Create agent (inject live client) ---
    print("\n[Phase 4] Initializing agent with reasoning-model-aware call sites...")
    agent = SelfCorrectingDataAgent(
        fs=fs,
        reader=reader,
        query=query_tool,
        stats=stats,
        transform=transform,
    )
    # Override the client with our long-timeout version
    agent.client = live_client

    # --- Phase 5: Run analyze() ---
    print("\n[Phase 5] Calling agent.analyze() -- waiting for LLM planning + synthesis...")
    print("-" * 70)

    start_time = time.time()
    try:
        result = agent.analyze(question, "/data/countries.csv")
        elapsed = time.time() - start_time

        # --- Phase 6: Display results ---
        print("-" * 70)
        print(f"\n[Phase 6] Results (total time: {elapsed:.1f}s)")
        print("=" * 70)

        print(f"\n  Question: {result.question}")
        print(f"  Dataset:  {result.dataset_path}")
        print(f"  Steps:    {len(result.steps)}")
        print(f"  Anomalies: {result.anomalies_found}")
        print(f"  Self-corrections: {result.self_corrections}")
        print(f"  Memory entries: {result.memory_entries_added}")

        print("\n  --- Execution Trace ---")
        for i, step in enumerate(result.steps):
            status = " [SELF-CORRECTED]" if step.corrected else ""
            anomalies_str = ""
            if step.anomalies:
                a_types = [f"{a.anomaly_type}({a.severity})" for a in step.anomalies]
                anomalies_str = f" [anomalies: {', '.join(a_types)}]"
            print(f"    Step {i+1}: {step.tool}{status}{anomalies_str}")
            print(f"      Reasoning: {step.reasoning}")
            print(f"      Result:    {step.result_summary}")

        print(f"\n  --- Final Answer ---")
        # Strip non-ASCII to avoid cp1252 UnicodeEncodeError on Windows
        safe_answer = result.final_answer.encode('ascii', 'replace').decode('ascii')
        print(f"  {safe_answer}")

        # Quality check
        has_content = len(result.final_answer) > 50
        has_numbers = any(c.isdigit() for c in result.final_answer)
        has_continents = any(c in result.final_answer.lower() for c in ["asia", "europe", "north america", "oceania", "south america", "africa"])
        
        print(f"\n  --- Quality Check ---")
        print(f"  Answer length: {len(result.final_answer)} chars {'[OK]' if has_content else '[SHORT]'}")
        print(f"  Contains numbers: {'[OK]' if has_numbers else '[MISSING]'}")
        print(f"  Mentions continents: {'[OK]' if has_continents else '[MISSING]'}")
        
        all_ok = has_content and has_numbers and has_continents
        print(f"\n{'=' * 70}")
        if all_ok:
            print(f"  [SUCCESS] Live LLM test completed in {elapsed:.1f}s -- answer is substantive!")
        else:
            print(f"  [PARTIAL] Live LLM test completed in {elapsed:.1f}s -- answer may lack detail")
        print(f"{'=' * 70}")
        return all_ok

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Live test failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# 4. Main
# =============================================================================

if __name__ == "__main__":
    questions = [
        # Simple question: less reasoning overhead, fits in 4096 tokens
        "What is the total GDP of all Asian countries? List each Asian country's GDP.",
    ]

    for q in questions:
        success = run_live_test(q, timeout=600)
        if not success:
            print("\nLive test did not complete successfully.")
            break
