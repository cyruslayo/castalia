"""Hardened demo: ReAct agent with live Tavily web search.

Addresses cold-endpoint latency by pre-warming the LLM, reduces
max_steps to 4, prints with flush, and adds JSON recovery for common
LLM formatting mistakes (e.g., tool name nested inside args).
"""
import sys
import time
sys.path.insert(0, ".")

from typing import Optional
from config import get_client, get_model
from tool_registry import ToolRegistry
from web_search_tools import register_tavily_tools
from agent_tool_integration import AdvancedToolAgent


class WebEnabledToolAgent(AdvancedToolAgent):
    """Agent subclass with explicit web-search guidance and JSON recovery."""

    def _build_system_prompt(self) -> str:
        base = super()._build_system_prompt()
        addition = (
            "\nCRITICAL RULES — READ CAREFULLY:\n"
            "1. Use web_search ONLY for recent events or current facts you do not know.\n"
            "2. You may call web_search at most TWO times total.\n"
            "3. After your FIRST search, if the results contain any relevant information, "
            "you MUST immediately provide a final answer with the 'answer' JSON format.\n"
            "4. Only do a second search if the first returned completely useless results.\n"
            "5. After TWO searches, you MUST answer — no more searching is allowed.\n"
            "6. When calling web_search, put 'tool' at the TOP level of the JSON, "
            "NOT inside the 'args' object.\n"
        )
        return base + addition

    def _extract_json(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response + scan ALL objects for a valid action.

        The LLM sometimes outputs multiple JSON blocks (e.g., a malformed
        first attempt followed by a correct one). We scan the entire text
        for all valid JSON objects, then pick the one that has 'tool'
        or 'answer' at the top level.
        """
        import json, re
        candidates = []

        # Strategy 1: try the full text as a single JSON object
        text_stripped = text.strip()
        if text_stripped.startswith("{"):
            try:
                candidates.append(json.loads(text_stripped))
            except json.JSONDecodeError:
                pass

        # Strategy 2: find every { ... } block via balanced-brace scan
        def _find_json_blocks(s: str):
            blocks = []
            i = 0
            while i < len(s):
                if s[i] == "{":
                    depth = 0
                    start = i
                    while i < len(s):
                        if s[i] == "{":
                            depth += 1
                        elif s[i] == "}":
                            depth -= 1
                            if depth == 0:
                                blocks.append(s[start:i + 1])
                                break
                        i += 1
                i += 1
            return blocks

        for block in _find_json_blocks(text):
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                pass

        # Strategy 3: markdown code fences
        for match in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
            for block in _find_json_blocks(match.group(1)):
                try:
                    candidates.append(json.loads(block))
                except json.JSONDecodeError:
                    pass

        if not candidates:
            return None

        # --- Pick the best candidate ---
        # Prefer objects with 'tool' or 'answer' at top level
        for c in candidates:
            if isinstance(c, dict) and ("tool" in c or "answer" in c):
                # Recovery: tool nested inside args
                if "args" in c and isinstance(c.get("args"), dict):
                    args = c["args"]
                    if "tool" in args and "tool" not in c:
                        c["tool"] = args.pop("tool")
                return c

        # Fallback: return the first valid dict
        for c in candidates:
            if isinstance(c, dict):
                return c

        return candidates[0] if candidates else None


def prewarm_llm():
    """Hit the vLLM endpoint once to reduce cold-start latency for real calls."""
    client = get_client()
    model = get_model()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return True
    except Exception as e:
        print("Prewarm failed: " + str(e))
        return False


def main():
    # Build minimal registry: only web_search to keep system prompt small
    registry = ToolRegistry()
    register_tavily_tools(registry)

    tool_names = sorted(registry.tools.keys())
    print("Registry loaded: " + str(len(tool_names)) + " tools", flush=True)
    for name in tool_names:
        print("  - " + name, flush=True)

    # Pre-warm (skipped: endpoint cold-start is 200-300s; first real call serves as warm-up)
    print("\nSkipping pre-warm (cold-start ~200s; first LLM call will warm the endpoint).", flush=True)

    # Agent with tight step budget (search + answer = 2 LLM calls)
    agent = WebEnabledToolAgent(registry, max_steps=6)

    query = (
        "What major AI model releases happened in April 2026? "
        "Search the web for current information, then provide your answer."
    )

    print("\nRunning agent (max 4 steps)...", flush=True)

    t0 = time.time()
    result = agent.run(query, verbose=True)
    elapsed = time.time() - t0

    # Results
    print("\n" + "=" * 60, flush=True)
    print("FINAL ANSWER", flush=True)
    print("=" * 60, flush=True)
    print(result["answer"], flush=True)

    print("\n" + "=" * 60, flush=True)
    print("METRICS", flush=True)
    print("=" * 60, flush=True)
    print("Total time:     {:.1f}s".format(elapsed), flush=True)
    print("Tool calls:     " + str(result["tool_calls"]), flush=True)
    print("Steps taken:    " + str(len(result["steps"])), flush=True)

    for step in result["steps"]:
        if step.get("type") == "tool":
            status = "OK" if step["result"].get("success") else "FAIL"
            print("  -> " + step["tool"] + " | " + status, flush=True)

    print("\nDemo complete.", flush=True)


if __name__ == "__main__":
    main()
