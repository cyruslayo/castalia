"""
EduTutor — Local Model Utilities

Shared helper module for loading Gemma 4 locally via Unsloth
instead of calling the Google API. All 4 notebooks import this.

Usage in any notebook:
    from local_model import load_teacher_model, generate_text, generate_json
"""
import torch
import json
import re
from pathlib import Path


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

_model = None
_tokenizer = None


def load_teacher_model(
    model_name: str = "google/gemma-4-e4b",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
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
    
    print(f"📦 Loading {model_name} (4-bit={load_in_4bit})...")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )
    
    FastLanguageModel.for_inference(_model)
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   Parameters: {_model.num_parameters():,}")
    return _model, _tokenizer


def load_finetuned_model(
    adapter_path: str,
    base_model: str = "google/gemma-4-e4b",
    max_seq_length: int = 4096,
):
    """Load a fine-tuned model with LoRA adapters.
    
    Used by Notebook 3 (eval) and Notebook 4 (demo) to load the EduTutor
    model after fine-tuning.
    """
    global _model, _tokenizer
    
    from unsloth import FastLanguageModel
    
    print(f"📦 Loading base model: {base_model}")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    
    print(f"🔌 Loading adapter from: {adapter_path}")
    from peft import PeftModel
    _model = PeftModel.from_pretrained(_model, adapter_path)
    
    FastLanguageModel.for_inference(_model)
    print(f"✅ Fine-tuned EduTutor loaded.")
    return _model, _tokenizer


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


def generate_json(
    prompt: str,
    max_new_tokens: int = 800,
    temperature: float = 0.1,
) -> dict | None:
    """Generate a JSON response from the locally loaded model.
    
    This is the drop-in replacement for API calls with:
        response_mime_type="application/json"
    
    It generates text, then extracts the first valid JSON object.
    """
    raw = generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    
    # Try to extract JSON from the response
    # Look for { ... } pattern
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try the whole response
    try:
        return json.loads(raw)
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
