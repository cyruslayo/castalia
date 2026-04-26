"""
EduTutor — Local Model Utilities

Shared helper module for loading Gemma 4 locally via Unsloth
instead of calling the Google API. All 4 notebooks import this.

Usage in any notebook:
    from local_model import load_teacher_model, generate_text, generate_json
    from local_model import EDUTUTOR_SYSTEM_PROMPT
"""
import torch
import json
import os
import re
from pathlib import Path


# ──────────────────────────────────────────────
# Shared System Prompt (Single Source of Truth)
# ──────────────────────────────────────────────

EDUTUTOR_SYSTEM_PROMPT = """You are EduTutor, a warm, patient, and neurodiversity-affirming AI tutor specializing in helping children aged 8-14 who learn differently. You are trained in evidence-based pedagogical strategies for ADHD, autism, dyslexia, and dyscalculia.

## Your Core Principles

1. **Strengths-Based:** You always presume competence. Every child can learn — they just need the right approach. You celebrate what they CAN do before addressing gaps.

2. **Productive Struggle:** You NEVER give answers directly. You guide, scaffold, and hint. You ask questions that lead the student to discover the answer themselves. If they get stuck, you break the problem into a smaller piece — but you never solve it for them.

3. **Cognitive Load Management:** You keep sentences short (under 15 words when possible). You use bullet points, numbered steps, and clear formatting. You present ONE idea at a time. You never give multi-step instructions all at once.

4. **Emotional Co-Regulation:** You constantly monitor the student's emotional tone. If they show signs of frustration (Yellow Zone), you pause teaching and validate their feelings first. If they're in crisis (Red Zone), you STOP all academic work and focus only on helping them feel safe. If they've shut down (Blue Zone), you gently re-engage with low-pressure activities.

5. **Adaptive Scaffolding:** You adjust your approach based on the student's profile:
   - **ADHD:** Keep it novel, use micro-chunks, celebrate small wins immediately, use movement analogies
   - **Autism:** Be explicit, provide structure, warn before transitions, connect to interests, avoid ambiguity
   - **Dyslexia:** Keep text minimal, use Orton-Gillingham phonics sequences, never force reading aloud, allow verbal responses
   - **Dyscalculia:** Use CRA (Concrete→Representational→Abstract), connect math to real objects, never time them

## Your Communication Style

- Warm but not patronizing
- Use the student's name when possible
- Short sentences, simple vocabulary
- Frequent check-ins: "How are you feeling about this?" / "Does that make sense so far?"
- Use emoji sparingly for encouragement: ⭐ 🎯 💪
- Format responses with clear visual structure (bullets, bold, numbered steps)

## What You Must NEVER Do

- Never say "This is easy" or "You should know this"
- Never give the full answer — always scaffold toward discovery
- Never give multi-step instructions all at once
- Never tell a student to "just focus" or "try harder"
- Never ignore emotional cues to push through content
- Never use sarcasm or irony — some students take things literally
- Never compare the student to others"""


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

_model = None
_tokenizer = None


def _resolve_cache_dir(cache_dir: str | Path | None = None) -> str | None:
    """Return the configured model cache directory, creating it when possible."""
    resolved = cache_dir or os.environ.get("EDUTUTOR_CACHE_DIR") or os.environ.get("HF_HOME")
    if resolved is None:
        return None

    path = Path(resolved)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_teacher_model(
    model_name: str = "google/gemma-4-e4b",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    cache_dir: str | Path | None = None,
):
    """Load a Gemma 4 model locally via Unsloth for data generation / judging.
    
    This replaces the Google API client. The same model is used as both
    the teacher (for data gen) and the judge (for eval), keeping everything
    local and API-key-free.
    
    Args:
        model_name: HuggingFace model ID. Default is Gemma 4 E4B.
                    For higher quality data gen, use "google/gemma-4-26b-a4b" if you have >40GB VRAM.
        max_seq_length: Context window. 4096 is enough for tutoring conversations.
        load_in_4bit: Use QLoRA quantization. Set False if you have enough VRAM.
        cache_dir: Optional HuggingFace/Unsloth cache directory. Defaults to
                   EDUTUTOR_CACHE_DIR or HF_HOME when set by the notebooks.
    
    Returns:
        (model, tokenizer) tuple
    """
    global _model, _tokenizer
    
    if _model is not None:
        print(f"✅ Model already loaded, reusing.")
        return _model, _tokenizer
    
    from unsloth import FastLanguageModel
    
    assert torch.cuda.is_available(), (
        "GPU required! Enable T4/A100 in Kaggle or use Colab Pro."
    )
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    if resolved_cache_dir:
        print(f"🗄️  Cache: {resolved_cache_dir}")
    
    print(f"📦 Loading {model_name} (4-bit={load_in_4bit})...")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
        cache_dir=resolved_cache_dir,
    )
    
    FastLanguageModel.for_inference(_model)
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   Parameters: {_model.num_parameters():,}")
    return _model, _tokenizer


def load_finetuned_model(
    adapter_path: str,
    base_model: str = "google/gemma-4-e4b",
    max_seq_length: int = 4096,
    cache_dir: str | Path | None = None,
):
    """Load a fine-tuned model with LoRA adapters via Unsloth's native loading.
    
    Used by Notebook 3 (eval) and Notebook 4 (demo) to load the EduTutor
    model after fine-tuning.
    
    This uses Unsloth's own adapter loading path rather than raw PeftModel,
    ensuring compatibility with FastLanguageModel.for_inference().
    """
    global _model, _tokenizer
    
    # Clear any previously loaded model to free VRAM
    if _model is not None:
        del _model
        _model = None
        torch.cuda.empty_cache()
    
    from unsloth import FastLanguageModel
    
    assert torch.cuda.is_available(), (
        "GPU required! Enable T4/A100 in Kaggle or use Colab Pro."
    )
    
    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    print(f"📦 Loading fine-tuned model from: {adapter_path}")
    print(f"   Base model: {base_model}")
    if resolved_cache_dir:
        print(f"   Cache: {resolved_cache_dir}")
    
    # Unsloth can load adapters directly via from_pretrained
    # by pointing model_name at the adapter directory
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        cache_dir=resolved_cache_dir,
    )
    
    FastLanguageModel.for_inference(_model)
    print(f"✅ Fine-tuned EduTutor loaded from {adapter_path}")
    return _model, _tokenizer


def unload_model():
    """Unload the current model to free VRAM for loading a different one."""
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
        torch.cuda.empty_cache()
        print("🗑️  Model unloaded, VRAM freed.")
    _tokenizer = None


# ──────────────────────────────────────────────
# Text Generation (replaces API calls)
# ──────────────────────────────────────────────

def generate_text(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_p: float = 0.95,
    system_prompt: str = None,
) -> str:
    """Generate text from the locally loaded model.
    
    This is the drop-in replacement for:
        client.aio.models.generate_content(model=MODEL, contents=prompt)
    
    Args:
        prompt: The user prompt / instruction.
        max_new_tokens: Max output length.
        temperature: Sampling temperature (higher = more diverse).
        top_p: Nucleus sampling threshold.
        system_prompt: Optional system prompt to prepend.
    
    Returns:
        Generated text string.
    """
    assert _model is not None, "Call load_teacher_model() first!"
    
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        outputs = _model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode only the new tokens (not the prompt)
    response = _tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    return response


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from text using balanced brace matching.
    
    Handles arbitrarily nested JSON (unlike a simple regex), which is
    required for the judge rubric's nested score objects.
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\":
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try the next opening brace
                    next_start = text.find("{", start + 1)
                    if next_start != -1:
                        return _extract_json(text[next_start:])
                    return None
    
    return None


def generate_json(
    prompt: str,
    max_new_tokens: int = 800,
    temperature: float = 0.1,
) -> dict | None:
    """Generate a JSON response from the locally loaded model.
    
    This is the drop-in replacement for API calls with:
        response_mime_type="application/json"
    
    It generates text, then extracts the first valid JSON object
    using balanced brace matching (supports arbitrary nesting depth).
    """
    raw = generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    
    # Use the robust balanced-brace extractor
    result = _extract_json(raw)
    if result is not None:
        return result
    
    # Fallback: try the whole response as JSON
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse JSON from response: {raw[:200]}...")
        return None


def generate_chat_response(
    messages: list[dict],
    max_new_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """Generate a response in a multi-turn conversation.
    
    Used by Notebook 4 for the agentic tutor's chat flow.
    
    Args:
        messages: List of {"role": "user"/"assistant"/"system", "content": "..."} dicts.
        max_new_tokens: Max output length.
        temperature: Sampling temperature.
    
    Returns:
        The assistant's response text.
    """
    assert _model is not None, "Call load_teacher_model() or load_finetuned_model() first!"
    
    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        outputs = _model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
    
    return _tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )


# ──────────────────────────────────────────────
# Convenience: get the current model / tokenizer
# ──────────────────────────────────────────────

def get_model():
    """Return the currently loaded model."""
    assert _model is not None, "No model loaded. Call load_teacher_model() first."
    return _model

def get_tokenizer():
    """Return the currently loaded tokenizer."""
    assert _tokenizer is not None, "No model loaded. Call load_teacher_model() first."
    return _tokenizer
