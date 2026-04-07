'use strict';

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_NOTEBOOK_METADATA = {
 colab: {
  provenance: [],
  gpuType: 'T4',
 },
 kernelspec: {
  display_name: 'Python 3',
  language: 'python',
  name: 'python3',
 },
 language_info: {
  name: 'python',
  version: '3.10.0',
 },
 accelerator: 'GPU',
};

const SETUP_BASIC_SOURCE = `# --- Google Colab Setup ---
!pip install -q "transformers>=4.51.0" accelerate bitsandbytes torch

import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

drive.mount('/content/drive')
CACHE_DIR = "/content/drive/MyDrive/models"
MODEL_NAME = "Qwen/Qwen3-14B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config,
    cache_dir=CACHE_DIR,
    torch_dtype="auto",
)

def generate(messages, max_new_tokens=512, temperature=0.7, do_sample=True, top_k=20):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print(f"✓ Model loaded: {MODEL_NAME}")
print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")`;

function isPlainObject(value) {
 return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function deepMerge(base, extra) {
 const output = Array.isArray(base) ? base.slice() : { ...base };

 if (!isPlainObject(extra)) {
  return output;
 }

 for (const [key, value] of Object.entries(extra)) {
  if (isPlainObject(value) && isPlainObject(output[key])) {
   output[key] = deepMerge(output[key], value);
   continue;
  }

  output[key] = Array.isArray(value) ? value.slice() : value;
 }

 return output;
}

function toSource(source = '') {
 if (Array.isArray(source)) {
  return source.map((line, index) => {
   const text = String(line).replace(/\r\n/g, '\n');
   return text.endsWith('\n') || index === source.length - 1 ? text : `${text}\n`;
  });
 }

 const normalized = String(source ?? '').replace(/\r\n/g, '\n');
 return normalized.match(/[^\n]*\n|[^\n]+/g) ?? [];
}

function markdown(source, metadata = {}) {
 return {
  cell_type: 'markdown',
  metadata: { ...metadata },
  source: toSource(source),
 };
}

function code(source, { metadata = {}, execution_count = null, outputs = [] } = {}) {
 return {
  cell_type: 'code',
  execution_count,
  metadata: { ...metadata },
  outputs: outputs.slice(),
  source: toSource(source),
 };
}

function cloneCell(cell) {
 return cell.cell_type === 'code'
  ? code(cell.source ?? [], {
    metadata: cell.metadata ?? {},
    execution_count: cell.execution_count ?? null,
    outputs: cell.outputs ?? [],
   })
  : markdown(cell.source ?? [], cell.metadata ?? {});
}

function notebook(cells, metadata = {}) {
 return {
  nbformat: 4,
  nbformat_minor: 0,
  metadata: deepMerge(DEFAULT_NOTEBOOK_METADATA, metadata),
  cells: cells.map(cloneCell),
 };
}

function save(filePath, cellsOrNotebook, metadata = {}) {
 const notebookJson = Array.isArray(cellsOrNotebook)
  ? notebook(cellsOrNotebook, metadata)
  : {
    ...cellsOrNotebook,
    metadata: deepMerge(DEFAULT_NOTEBOOK_METADATA, cellsOrNotebook.metadata ?? {}),
    cells: (cellsOrNotebook.cells ?? []).map(cloneCell),
   };

 const resolvedPath = path.resolve(filePath);
 fs.mkdirSync(path.dirname(resolvedPath), { recursive: true });
 fs.writeFileSync(resolvedPath, `${JSON.stringify(notebookJson, null, 1)}\n`, 'utf8');
 return resolvedPath;
}

const SETUP_BASIC = code(SETUP_BASIC_SOURCE);

module.exports = {
 DEFAULT_NOTEBOOK_METADATA,
 NOTEBOOK_METADATA: DEFAULT_NOTEBOOK_METADATA,
 SETUP_BASIC_SOURCE,
 SETUP_BASIC,
 toSource,
 md: markdown,
 markdown,
 code,
 notebook,
 save,
};
