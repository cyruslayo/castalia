"""
LLM Configuration - Connection settings for the model.

This keeps credentials and settings in one place so the rest of the
code stays clean.
"""

import os
from pathlib import Path

from openai import OpenAI


def _load_dotenv() -> None:
    """Lightweight .env loader so project secrets work without extra deps."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# --- Connection settings ---
# This is an OpenAI-compatible endpoint (vLLM server)
LLM_CONFIG = {
    "base_url": "https://layoayeni--hermes-openai-compatible-vllm-serve.modal.run/v1",
    "model": "hermes-model",
    "api_key": "Zwu0tLZf9x2qqqd4vttgm54qQADehVBgr__ekQI0iAA",
}

# Create a reusable client instance
# The OpenAI client handles connection pooling, so we create it once
client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
)


def get_client() -> OpenAI:
    """Return the configured OpenAI-compatible client."""
    return client

def get_model() -> str:
    """Return the configured model name."""
    return LLM_CONFIG["model"]


# --- Tavily Configuration ---
TAVILY_CONFIG = {
    "base_url": "https://api.tavily.com",
}


def get_tavily_api_key() -> str:
    """Return the Tavily API key, preferring env/.env at call time."""
    return os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY_HERE")


def get_tavily_base_url() -> str:
    """Return the Tavily API base URL."""
    return os.environ.get("TAVILY_BASE_URL", TAVILY_CONFIG["base_url"])
