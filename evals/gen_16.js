'use strict';

const { md, code, save, SETUP_BASIC } = require('./gen');

const cells = [
 md(`# Notebook 16 — Cost, Latency, and Reliability Eval

Agent quality is only one part of operational performance. In this notebook you will build a first-principles evaluation harness for cost, latency, timeouts, retries, and reliability so you can compare agent policies with explicit trade-offs instead of anecdotes.`),

 md(`## What you will build

- a reproducible benchmark set for several agent tasks
- a synthetic but realistic request simulator with timeouts and retries
- experiment summaries for quality vs latency and quality vs cost
- small notebook dashboards for timeout, retry, and reliability trade-offs
- artifact files you can reuse in later regression and reporting notebooks`),

 md(`## Design principle

For teaching, we start with transparent telemetry you can inspect end to end. That keeps the notebook runnable and understandable. At the end, an optional live probe measures the open-source local model loaded by \`SETUP_BASIC\` so the same framework can score real runs too.`),

 SETUP_BASIC,

 md(`## Step 1: Add notebook helpers and artifact paths

We will reuse standard-library utilities only. The helpers below generate markdown tables, percentile summaries, and lightweight ASCII dashboards.`),

 code(`from collections import Counter, defaultdict

random.seed(16)

ARTIFACT_DIR = Path("artifacts") / "notebook_16_cost_latency_reliability"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def to_markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\\n".join([header, divider, *body])

def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

def clipped(value, low=0.0, high=1.0):
    return max(low, min(high, value))

def ascii_bar(value, width=16, filled="█"):
    value = clipped(value)
    filled_count = int(round(value * width))
    return filled * filled_count + "·" * (width - filled_count)

print("Artifact directory:", ARTIFACT_DIR.resolve())`),

 md(`## Step 2: Define the benchmark tasks and system profiles

Each benchmark case represents a task the agent must complete. We keep a few fields that strongly affect operations work:

- difficulty
- prompt and completion token budgets
- tool hops
- minimum acceptable score for the request to count as successful`),

 code(`benchmark_cases = [
    {
        "id": "route_password_reset",
        "family": "tool_routing",
        "difficulty": 0.9,
        "prompt_tokens": 180,
        "output_tokens": 60,
        "tool_hops": 1,
        "success_threshold": 1.0,
    },
    {
        "id": "incident_summary",
        "family": "summarization",
        "difficulty": 1.0,
        "prompt_tokens": 260,
        "output_tokens": 120,
        "tool_hops": 1,
        "success_threshold": 0.6,
    },
    {
        "id": "policy_exception_check",
        "family": "policy",
        "difficulty": 1.15,
        "prompt_tokens": 240,
        "output_tokens": 90,
        "tool_hops": 1,
        "success_threshold": 1.0,
    },
    {
        "id": "rollback_plan",
        "family": "planning",
        "difficulty": 1.25,
        "prompt_tokens": 320,
        "output_tokens": 180,
        "tool_hops": 2,
        "success_threshold": 0.6,
    },
    {
        "id": "log_triage",
        "family": "analysis",
        "difficulty": 1.35,
        "prompt_tokens": 340,
        "output_tokens": 160,
        "tool_hops": 2,
        "success_threshold": 0.6,
    },
    {
        "id": "customer_comms",
        "family": "customer_comms",
        "difficulty": 0.95,
        "prompt_tokens": 210,
        "output_tokens": 110,
        "tool_hops": 1,
        "success_threshold": 0.6,
    },
    {
        "id": "security_triage",
        "family": "security",
        "difficulty": 1.45,
        "prompt_tokens": 360,
        "output_tokens": 150,
        "tool_hops": 3,
        "success_threshold": 1.0,
    },
    {
        "id": "tool_repair_plan",
        "family": "agent_recovery",
        "difficulty": 1.55,
        "prompt_tokens": 380,
        "output_tokens": 200,
        "tool_hops": 3,
        "success_threshold": 0.6,
    },
]

system_profiles = {
    "small-fast": {
        "name": "small-fast",
        "base_quality": 0.71,
        "base_latency_s": 1.45,
        "latency_jitter": 0.24,
        "input_cost_per_1k": 0.0012,
        "output_cost_per_1k": 0.0018,
    },
    "balanced": {
        "name": "balanced",
        "base_quality": 0.81,
        "base_latency_s": 2.45,
        "latency_jitter": 0.22,
        "input_cost_per_1k": 0.0022,
        "output_cost_per_1k": 0.0032,
    },
    "careful-large": {
        "name": "careful-large",
        "base_quality": 0.9,
        "base_latency_s": 4.2,
        "latency_jitter": 0.2,
        "input_cost_per_1k": 0.004,
        "output_cost_per_1k": 0.006,
    },
}

print("Benchmark cases:", len(benchmark_cases))
print("System profiles:", list(system_profiles))`),

 md(`## Step 3: Simulate one request attempt

The simulator keeps four ideas separate:

1. raw latency before any timeout
2. observed latency after timeout truncation
3. quality of the returned answer
4. cost billed for prompt and completion tokens

This gives us enough structure to reason about reliability trade-offs without needing a proprietary eval platform.`),

 code(`def score_outcome(quality_signal, case, rng):
    partial_band = clipped(0.2 + 0.08 * case["difficulty"] - 0.1 * quality_signal, 0.08, 0.34)
    roll = rng.random()
    if roll < quality_signal:
        return "correct", 1.0
    if roll < quality_signal + partial_band:
        return "partial", 0.6
    return "wrong", 0.0

def sample_attempt(case, profile, timeout_s, retry_index, run_index):
    rng = random.Random(f'{case["id"]}|{profile["name"]}|{timeout_s}|{retry_index}|{run_index}')
    latency_center = profile["base_latency_s"] * case["difficulty"] * (1 + 0.15 * case["tool_hops"]) * (0.92 ** retry_index)
    raw_latency_s = rng.lognormvariate(math.log(max(latency_center, 0.05)), profile["latency_jitter"])
    timeout = raw_latency_s > timeout_s
    observed_latency_s = timeout_s if timeout else raw_latency_s

    prompt_tokens = case["prompt_tokens"] + retry_index * 32
    completion_tokens = case["output_tokens"] + case["tool_hops"] * 18
    billed_completion_tokens = int(completion_tokens * (0.35 if timeout else 1.0))
    cost_usd = (prompt_tokens / 1000) * profile["input_cost_per_1k"] + (billed_completion_tokens / 1000) * profile["output_cost_per_1k"]

    quality_signal = clipped(
        profile["base_quality"]
        - 0.055 * (case["difficulty"] - 1.0)
        + 0.03 * retry_index
        - (0.06 if timeout else 0.0),
        0.0,
        0.97,
    )

    if timeout:
        outcome, score = "timeout", 0.0
    else:
        outcome, score = score_outcome(quality_signal, case, rng)

    return {
        "outcome": outcome,
        "score": score,
        "timeout": timeout,
        "quality_signal": round(quality_signal, 3),
        "raw_latency_s": round(raw_latency_s, 3),
        "observed_latency_s": round(observed_latency_s, 3),
        "cost_usd": round(cost_usd, 5),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": billed_completion_tokens,
    }

def execute_policy(case, profile, timeout_s, max_retries, run_index):
    attempts = []
    for retry_index in range(max_retries + 1):
        attempt = sample_attempt(case, profile, timeout_s, retry_index, run_index)
        attempts.append(attempt)
        if attempt["score"] >= case["success_threshold"]:
            break

    total_latency_s = sum(attempt["observed_latency_s"] for attempt in attempts) + 0.25 * max(0, len(attempts) - 1)
    total_cost_usd = sum(attempt["cost_usd"] for attempt in attempts)
    final_attempt = attempts[-1]
    initial_attempt = attempts[0]

    return {
        "case_id": case["id"],
        "family": case["family"],
        "profile": profile["name"],
        "timeout_s": timeout_s,
        "max_retries": max_retries,
        "policy_id": f'{profile["name"]}|t{timeout_s}|r{max_retries}',
        "attempts": len(attempts),
        "timeout_count": sum(1 for attempt in attempts if attempt["timeout"]),
        "total_latency_s": round(total_latency_s, 3),
        "total_cost_usd": round(total_cost_usd, 5),
        "quality_score": final_attempt["score"],
        "final_outcome": final_attempt["outcome"],
        "success": final_attempt["score"] >= case["success_threshold"],
        "used_retry": len(attempts) > 1,
        "improved_after_retry": len(attempts) > 1 and initial_attempt["score"] < case["success_threshold"] and final_attempt["score"] >= case["success_threshold"],
    }`),

 md(`## Step 4: Inspect a few deterministic request traces

Before running a full benchmark, inspect single requests. This makes it clear what the simulator is doing when latency, cost, and retries accumulate.`),

 code(`trace_rows = []
for profile_name in system_profiles:
    case = benchmark_cases[0]
    profile = system_profiles[profile_name]
    run = execute_policy(case, profile, timeout_s=3.5, max_retries=1, run_index=0)
    trace_rows.append(
        {
            "policy_id": run["policy_id"],
            "case_id": run["case_id"],
            "attempts": run["attempts"],
            "final_outcome": run["final_outcome"],
            "quality_score": run["quality_score"],
            "total_latency_s": run["total_latency_s"],
            "total_cost_usd": run["total_cost_usd"],
        }
    )

print(to_markdown_table(trace_rows, ["policy_id", "case_id", "attempts", "final_outcome", "quality_score", "total_latency_s", "total_cost_usd"]))`),

 md(`## Step 5: Build the experiment grid

We will compare each profile under multiple timeout budgets and retry caps. That lets us isolate the shape of the trade-off instead of blaming a single average number.`),

 code(`timeout_options = [2.5, 4.0, 6.0]
retry_options = [0, 1, 2]
repeats_per_case = 18

policy_grid = []
for profile_name, profile in system_profiles.items():
    for timeout_s in timeout_options:
        for max_retries in retry_options:
            policy_grid.append(
                {
                    "profile": profile_name,
                    "timeout_s": timeout_s,
                    "max_retries": max_retries,
                    "policy_id": f'{profile_name}|t{timeout_s}|r{max_retries}',
                }
            )

experiment_rows = []
for policy in policy_grid:
    profile = system_profiles[policy["profile"]]
    for case in benchmark_cases:
        for run_index in range(repeats_per_case):
            experiment_rows.append(
                execute_policy(
                    case,
                    profile,
                    timeout_s=policy["timeout_s"],
                    max_retries=policy["max_retries"],
                    run_index=run_index,
                )
            )

print("Policies:", len(policy_grid))
print("Experiment rows:", len(experiment_rows))`),

 md(`## Step 6: Aggregate the benchmark metrics

From request-level logs we derive the operational metrics we actually care about:

- mean quality
- request success rate
- timeout rate
- retry rate
- mean and p95 latency
- mean cost and cost per successful request`),

 code(`def summarize_policy(rows):
    success_rate = statistics.fmean(1 if row["success"] else 0 for row in rows)
    timeout_rate = statistics.fmean(1 if row["timeout_count"] > 0 else 0 for row in rows)
    retry_rate = statistics.fmean(1 if row["used_retry"] else 0 for row in rows)
    mean_quality = statistics.fmean(row["quality_score"] for row in rows)
    mean_cost = statistics.fmean(row["total_cost_usd"] for row in rows)
    mean_latency = statistics.fmean(row["total_latency_s"] for row in rows)
    successes = sum(1 for row in rows if row["success"])
    cost_per_success = sum(row["total_cost_usd"] for row in rows) / max(successes, 1)
    retry_salvage = sum(1 for row in rows if row["improved_after_retry"]) / max(sum(1 for row in rows if row["used_retry"]), 1)
    reliability_index = 0.55 * success_rate + 0.25 * (1 - timeout_rate) + 0.2 * (1 - retry_rate)

    return {
        "policy_id": rows[0]["policy_id"],
        "profile": rows[0]["profile"],
        "timeout_s": rows[0]["timeout_s"],
        "max_retries": rows[0]["max_retries"],
        "mean_quality": round(mean_quality, 3),
        "success_rate": round(success_rate, 3),
        "timeout_rate": round(timeout_rate, 3),
        "retry_rate": round(retry_rate, 3),
        "mean_attempts": round(statistics.fmean(row["attempts"] for row in rows), 2),
        "mean_latency_s": round(mean_latency, 3),
        "p95_latency_s": round(percentile([row["total_latency_s"] for row in rows], 95), 3),
        "mean_cost_usd": round(mean_cost, 4),
        "cost_per_success_usd": round(cost_per_success, 4),
        "retry_salvage_rate": round(retry_salvage, 3),
        "reliability_index": round(reliability_index, 3),
    }

rows_by_policy = defaultdict(list)
for row in experiment_rows:
    rows_by_policy[row["policy_id"]].append(row)

summary_rows = [summarize_policy(rows) for rows in rows_by_policy.values()]
summary_rows = sorted(summary_rows, key=lambda row: (-row["success_rate"], row["mean_latency_s"], row["mean_cost_usd"]))

print(to_markdown_table(summary_rows[:9], ["policy_id", "mean_quality", "success_rate", "timeout_rate", "mean_latency_s", "p95_latency_s", "mean_cost_usd", "retry_rate"]))`),

 md(`## Step 7: Build a quality-vs-latency dashboard

A system can improve quality and still feel worse to users if latency expands too much. This dashboard keeps the latency trade-off visible.`),

 code(`latency_dashboard_rows = sorted(summary_rows, key=lambda row: (row["mean_latency_s"], -row["mean_quality"]))[:9]
view_rows = []
for row in latency_dashboard_rows:
    latency_ratio = min(row["mean_latency_s"] / 8.0, 1.0)
    view_rows.append(
        {
            "policy_id": row["policy_id"],
            "quality": row["mean_quality"],
            "latency_s": row["mean_latency_s"],
            "quality_bar": ascii_bar(row["mean_quality"], width=14),
            "latency_bar": ascii_bar(latency_ratio, width=14, filled="▓"),
        }
    )

print(to_markdown_table(view_rows, ["policy_id", "quality", "latency_s", "quality_bar", "latency_bar"]))`),

 md(`## Step 8: Build a quality-vs-cost dashboard

Cost matters per request, but cost per successful request matters even more when failures and retries force more work.`),

 code(`def pareto_frontier(rows, x_key, y_key):
    frontier = []
    best_y = -1.0
    for row in sorted(rows, key=lambda item: (item[x_key], -item[y_key])):
        if row[y_key] > best_y:
            frontier.append(row)
            best_y = row[y_key]
    return frontier

quality_cost_frontier = pareto_frontier(summary_rows, "mean_cost_usd", "mean_quality")
frontier_rows = []
for row in quality_cost_frontier:
    frontier_rows.append(
        {
            "policy_id": row["policy_id"],
            "mean_cost_usd": row["mean_cost_usd"],
            "mean_quality": row["mean_quality"],
            "cost_per_success_usd": row["cost_per_success_usd"],
            "efficiency": round(row["mean_quality"] / row["mean_cost_usd"], 1),
        }
    )

print(to_markdown_table(frontier_rows, ["policy_id", "mean_cost_usd", "mean_quality", "cost_per_success_usd", "efficiency"]))`),

 md(`## Step 9: Analyze timeout trade-offs directly

Timeouts are one of the cleanest examples of reliability engineering:

- shorter budgets reduce tail latency
- shorter budgets also increase dropped requests and trigger more retries
- longer budgets may improve success while hurting responsiveness`),

 code(`timeout_rows = []
for profile_name in system_profiles:
    for timeout_s in timeout_options:
        row = next(item for item in summary_rows if item["profile"] == profile_name and item["timeout_s"] == timeout_s and item["max_retries"] == 1)
        timeout_rows.append(
            {
                "profile": profile_name,
                "timeout_s": timeout_s,
                "success_rate": row["success_rate"],
                "timeout_rate": row["timeout_rate"],
                "mean_latency_s": row["mean_latency_s"],
                "p95_latency_s": row["p95_latency_s"],
            }
        )

print(to_markdown_table(timeout_rows, ["profile", "timeout_s", "success_rate", "timeout_rate", "mean_latency_s", "p95_latency_s"]))`),

 md(`## Step 10: Measure retry behavior

Retries are not free. They can recover failed requests, but they also raise latency and cost. We want to know when the extra reliability is worth it.`),

 code(`retry_rows = []
for profile_name in system_profiles:
    baseline_row = next(item for item in summary_rows if item["profile"] == profile_name and item["timeout_s"] == 4.0 and item["max_retries"] == 0)
    for max_retries in retry_options:
        row = next(item for item in summary_rows if item["profile"] == profile_name and item["timeout_s"] == 4.0 and item["max_retries"] == max_retries)
        retry_rows.append(
            {
                "profile": profile_name,
                "max_retries": max_retries,
                "success_rate": row["success_rate"],
                "delta_success_vs_r0": round(row["success_rate"] - baseline_row["success_rate"], 3),
                "delta_latency_vs_r0": round(row["mean_latency_s"] - baseline_row["mean_latency_s"], 3),
                "delta_cost_vs_r0": round(row["mean_cost_usd"] - baseline_row["mean_cost_usd"], 4),
                "retry_salvage_rate": row["retry_salvage_rate"],
            }
        )

print(to_markdown_table(retry_rows, ["profile", "max_retries", "success_rate", "delta_success_vs_r0", "delta_latency_vs_r0", "delta_cost_vs_r0", "retry_salvage_rate"]))`),

 md(`## Step 11: Build a small reliability dashboard

This dashboard turns the benchmark into an operator-facing view. High bars are better for quality, success, and reliability. Lower latency and lower cost should be interpreted alongside those bars.`),

 code(`dashboard_rows = []
for row in summary_rows[:8]:
    dashboard_rows.append(
        {
            "policy_id": row["policy_id"],
            "quality": ascii_bar(row["mean_quality"], width=12),
            "success": ascii_bar(row["success_rate"], width=12),
            "reliability": ascii_bar(row["reliability_index"], width=12),
            "latency_s": row["mean_latency_s"],
            "cost_usd": row["mean_cost_usd"],
        }
    )

print(to_markdown_table(dashboard_rows, ["policy_id", "quality", "success", "reliability", "latency_s", "cost_usd"]))`),

 md(`## Step 12: Turn metrics into service-level gates

Evaluation becomes operational when it answers a shipping question. Here we create a simple SLO gate for candidate policies.`),

 code(`SLO = {
    "min_success_rate": 0.82,
    "min_mean_quality": 0.72,
    "max_timeout_rate": 0.2,
    "max_p95_latency_s": 7.5,
    "max_mean_cost_usd": 0.0075,
}

release_rows = []
for row in summary_rows:
    passed = (
        row["success_rate"] >= SLO["min_success_rate"]
        and row["mean_quality"] >= SLO["min_mean_quality"]
        and row["timeout_rate"] <= SLO["max_timeout_rate"]
        and row["p95_latency_s"] <= SLO["max_p95_latency_s"]
        and row["mean_cost_usd"] <= SLO["max_mean_cost_usd"]
    )
    release_rows.append(
        {
            "policy_id": row["policy_id"],
            "release_ready": passed,
            "success_rate": row["success_rate"],
            "p95_latency_s": row["p95_latency_s"],
            "mean_cost_usd": row["mean_cost_usd"],
            "timeout_rate": row["timeout_rate"],
        }
    )

print(to_markdown_table(release_rows[:10], ["policy_id", "release_ready", "success_rate", "p95_latency_s", "mean_cost_usd", "timeout_rate"]))`),

 md(`## Step 13: Write an experiment summary

A good evaluation notebook ends with a short operational report: which policy should be default, which should be fallback, and what trade-off deserves monitoring.`),

 code(`release_candidates = [row for row in summary_rows if next(item for item in release_rows if item["policy_id"] == row["policy_id"])["release_ready"]]
default_policy = release_candidates[0] if release_candidates else summary_rows[0]
fallback_policy = sorted(summary_rows, key=lambda row: (-row["success_rate"], row["mean_latency_s"]))[0]

report = {
    "default_policy": default_policy["policy_id"],
    "fallback_policy": fallback_policy["policy_id"],
    "best_quality_cost_frontier": [row["policy_id"] for row in quality_cost_frontier[:4]],
    "watch_items": [],
}

if default_policy["retry_rate"] > 0.45:
    report["watch_items"].append("Default policy depends heavily on retries; investigate first-attempt quality or routing.")
if fallback_policy["timeout_rate"] > 0.2:
    report["watch_items"].append("Fallback path still times out too often under the current timeout budget.")
if quality_cost_frontier[0]["profile"] == "small-fast":
    report["watch_items"].append("The cheapest frontier point is small-fast; use it only for lower-risk traffic classes.")

print(json.dumps(report, indent=2))`),

 md(`## Step 14: Optional live local-model probe

The synthetic benchmark keeps the trade-offs transparent. When you want to connect the notebook to the real open-source model from \`SETUP_BASIC\`, run the probe below. It measures actual wall-clock latency and token counts on a few short prompts.`),

 code(`RUN_LIVE_PROBE = False

probe_prompts = [
    "Classify the ticket as billing, technical, or account: 'Customer cannot reset password after clicking the email link.'",
    "Summarize this incident in one sentence: Database failover caused latency spikes. The team rolled back within twelve minutes.",
    "Explain whether copying customer data to a personal USB drive is allowed under a typical enterprise policy.",
]

live_probe_rows = []
if RUN_LIVE_PROBE:
    for prompt in probe_prompts:
        started = time.perf_counter()
        answer = generate(prompt, max_new_tokens=96, temperature=0.0, do_sample=False)
        latency_s = time.perf_counter() - started
        prompt_tokens = len(tokenizer.encode(prompt))
        answer_tokens = len(tokenizer.encode(answer))
        live_probe_rows.append(
            {
                "prompt_preview": prompt[:42] + ("..." if len(prompt) > 42 else ""),
                "latency_s": round(latency_s, 3),
                "prompt_tokens": prompt_tokens,
                "answer_tokens": answer_tokens,
            }
        )
    print(to_markdown_table(live_probe_rows, ["prompt_preview", "latency_s", "prompt_tokens", "answer_tokens"]))
else:
    print("Set RUN_LIVE_PROBE = True to run the local-model latency probe.")`),

 md(`## Step 15: Save benchmark artifacts

We save both the detailed experiment rows and the condensed summaries. That gives later notebooks reproducible inputs for regression tests and reporting.`),

 code(`summary_path = ARTIFACT_DIR / "policy_summary.json"
frontier_path = ARTIFACT_DIR / "quality_cost_frontier.json"
report_path = ARTIFACT_DIR / "ops_report.json"
detail_path = ARTIFACT_DIR / "experiment_rows_sample.json"
csv_path = ARTIFACT_DIR / "policy_summary.csv"

summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
frontier_path.write_text(json.dumps(frontier_rows, indent=2), encoding="utf-8")
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
detail_path.write_text(json.dumps(experiment_rows[:80], indent=2), encoding="utf-8")

with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print("Saved:", summary_path)
print("Saved:", frontier_path)
print("Saved:", report_path)
print("Saved:", detail_path)
print("Saved:", csv_path)`),

 md(`## Recap

You now have a notebook-native evaluation workflow for:

- quality vs latency
- quality vs cost
- timeout analysis
- retry analysis
- reliability-aware release gates

That is the pattern you want for real agent systems: define the operational budget, run controlled experiments, inspect the trade-offs, and only then choose the default policy.`),
];

const outputPath = save('16_cost_latency_reliability_eval.ipynb', cells);
console.log(`Generated ${outputPath}`);
