# Castalia — Notebook Style Guide

## Purpose

Every Castalia notebook should read like a chapter in a textbook **and** run cleanly in Google Colab Pro. A learner should be able to open any notebook, understand its place in the curriculum, learn the concepts from first principles, run every cell, practice with exercises, and know where to go next.

---

## Notebook Structure Template

Every notebook MUST follow this structure:

### 1. Title Cell (markdown)
```markdown
# [Number] — [Title]

> **Orbit [N]** · **Domain**: [PE/RAG/AG/EV/FT/MM/SY] · **Difficulty**: [Beginner/Intermediate/Advanced] · **Reading time**: ~[N] min

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[OWNER]/castalia/blob/main/[path/to/notebook.ipynb])

**Prerequisites**: [Notebook links]
- [previous_notebook.ipynb](../path/to/previous_notebook.ipynb) — [one-line description]

**What you will learn**:
- [Learning objective 1]
- [Learning objective 2]
- [Learning objective 3]
```

### 2. Setup Cell (code)
```python
# @title Setup — Run this cell first
# --- Google Colab Setup ---
!pip install -q [domain-specific packages]

import ...
```

### 3–N. Content Cells (alternating markdown + code)
- Lead every code section with a markdown cell explaining **why** before **how**
- Use `## Section Number — Section Title` headings
- Include ASCII or matplotlib diagrams for key concepts

### N-3. Exercises Cell (markdown + code)
```markdown
## Exercises

1. **[Exercise Title]** — [description of what to build/modify]
2. **[Exercise Title]** — [description]
```

### N-2. Key Takeaways Cell (markdown)
```markdown
## Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | [Concise insight] |
| 2 | [Concise insight] |
```

### N-1. What's Next Cell (markdown)
```markdown
## What's Next

- **Next**: [next_notebook.ipynb](../path) — [description]
- **Related**: [related_notebook.ipynb](../path) — [description]
```

### N. References Cell (markdown)
```markdown
## References & Further Reading

1. [Author(s), "Title," Year](URL) — [one-line annotation]
2. [Documentation link](URL) — [annotation]
```

---

## Prose Style

- **First principles first**: Explain *why* something exists before showing *how* to use it
- **Derive, don't just demonstrate**: If a formula or algorithm is used, walk through its derivation or intuition
- **Tables over walls of text**: Use comparison tables for tradeoffs, decision matrices, feature comparisons
- **Name the failure modes**: Every technique section should include "when this fails" or "limitations"
- **Use concrete numbers**: "This takes ~2.8 GB VRAM" not "this uses some memory"
- **Sentence case for headings**: "Building a retrieval pipeline" not "Building A Retrieval Pipeline"
- **Active voice**: "The embedder maps text to vectors" not "Text is mapped to vectors by the embedder"

---

## Code Style

- **No framework magic**: Build core concepts from scratch before showing library shortcuts
- **Type hints**: Use Python type hints in function signatures
- **Docstrings**: Every function > 5 lines gets a one-line docstring
- **Print intermediate results**: Show the learner what's happening at each step
- **Minimal dependencies**: Use standard library + domain essentials. Never install a library for one line of code
- **Version pinning**: Use `pip install -q "transformers>=4.51.0"` — pin minimum versions for reproducibility

---

## Colab Requirements

- **GPU metadata**: Every notebook must have `"accelerator": "GPU"` and `"gpuType": "T4"` in metadata
- **Drive caching**: Cache large model downloads to `/content/drive/MyDrive/models`
- **Memory awareness**: Print VRAM usage after model loading. Warn if a cell requires >15 GB
- **Cell independence**: After the setup cell, each section should be re-runnable without re-running all prior cells (where practical)

---

## Quality Bar

A notebook is ready when:
- [ ] Title cell has orbit, difficulty, reading time, Colab badge, prerequisites, learning objectives
- [ ] Setup cell runs without errors on Colab Pro (T4 GPU)
- [ ] Every concept is explained before its code
- [ ] At least 2 exercises with clear instructions
- [ ] Key takeaways table present
- [ ] Cross-references to next and related notebooks
- [ ] References section with at least 2 external sources
- [ ] Markdown-to-code ratio between 0.8:1 and 1.5:1
- [ ] No empty cells, no TODO comments, no placeholder text
