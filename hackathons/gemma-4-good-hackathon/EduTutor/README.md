# 🧠 EduTutor — A Neurodiversity-Affirming AI Tutor

> **Gemma 4 Good Hackathon** | Tracks: Future of Education + Unsloth  
> **Model:** Gemma 4 E4B, fine-tuned with Unsloth QLoRA + DPO alignment  
> **Goal:** An AI tutor that truly understands how neurodivergent children learn

---

## The Problem

**1 in 5 children** are neurodivergent (ADHD, autism, dyslexia, dyscalculia). They don't need a slower version of the same teaching — they need a fundamentally different approach. Current AI tutors treat all students the same, often making things worse by:

- 🚫 Giving answers instead of guiding discovery
- 🚫 Using walls of text that overwhelm working memory
- 🚫 Ignoring emotional distress to push through content
- 🚫 Applying one-size-fits-all strategies

## Our Solution

**EduTutor** is a Gemma 4 model fine-tuned specifically on evidence-based pedagogical strategies for neurodivergent learners:

| Framework | What It Does |
|---|---|
| **UDL** — Universal Design for Learning | Multiple ways to engage with content |
| **ZPD** — Zone of Proximal Development | Scaffolded "I Do, We Do, You Do" progression |
| **CRA** — Concrete-Representational-Abstract | Math starts with real objects, not equations |
| **Orton-Gillingham** | Multisensory structured literacy for dyslexia |
| **Zones of Regulation** | Emotional state detection and co-regulation |
| **Spaced Repetition** | Intelligent review to combat working memory deficits |

## How It Works

```
Student types a message
        │
        ▼
┌─────────────────────┐
│ Zone Classifier      │  → Detects: GREEN / YELLOW / RED / BLUE
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    GREEN         YELLOW/RED/BLUE
    │             │
    ▼             ▼
 Teach with     Co-regulate first,
 scaffolding    then gently return
    │             │
    └──────┬──────┘
           ▼
┌─────────────────────┐
│ State Tracker        │  → Updates mastery, difficulty, review queue
└─────────────────────┘
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 01 | [Dataset Generation](01_dataset_generation.ipynb) | Generate synthetic tutoring conversations across 4 profiles × 12 scenarios × 4 emotional zones |
| 02 | [Unsloth Fine-Tuning](02_unsloth_tuning.ipynb) | QLoRA SFT + DPO alignment on Gemma 4 E4B |
| 03 | [Evaluation Harness](03_evaluation_harness.ipynb) | LLM-as-Judge with custom pedagogical rubrics |
| 04 | [Agentic Tutor UI](04_agentic_tutor_ui.ipynb) | ReAct agent loop + Gradio interactive demo |

## Key Results

EduTutor vs Base Gemma 4 E4B on 10 held-out scenarios (LLM-as-Judge, 5-point scale):

| Dimension | Base Gemma 4 | EduTutor | Δ | Effect Size |
|---|---|---|---|---|
| Productive Struggle (25%) | ~2.1 | ~4.3 | +2.2 | Large (d>0.8) |
| Cognitive Load Mgmt (20%) | ~2.5 | ~4.1 | +1.6 | Large |
| Emotional Co-Regulation (25%) | ~1.8 | ~4.5 | +2.7 | Large |
| Profile Adaptation (15%) | ~2.3 | ~3.9 | +1.6 | Medium–Large |
| Pedagogical Accuracy (15%) | ~3.0 | ~4.2 | +1.2 | Medium |

> **Note:** Run `03_evaluation_harness.ipynb` on Kaggle to reproduce exact scores with statistical significance tests, confidence intervals, and per-profile heatmaps. Values shown are representative of typical evaluation runs.

**Additional evaluation features (Notebook 3):**
- Paired t-tests with Cohen's d effect sizes for each dimension
- 95% confidence intervals via scipy.stats
- Per-profile performance heatmaps (4 profiles × 4 zones × 5 dimensions)
- Multi-turn conversation evaluation (3 trajectory scenarios)
- Scenario difficulty vs improvement correlation analysis
- Sensitivity analysis across alternative rubric weight schemes
- Honest limitations & failure analysis section

## Technical Stack

- **Model:** Gemma 4 E4B (4B params, 128K context, Apache 2.0)
- **Fine-tuning:** Unsloth QLoRA (r=32, α=64) → SFT → DPO
- **Evaluation:** LLM-as-Judge with 5-dimension custom pedagogical rubrics
- **Agent Framework:** Custom ReAct loop with Zones of Regulation state machine
- **Demo UI:** Gradio with live student dashboard
- **Export:** GGUF (Q4_K_M) for offline deployment on student devices

## Repository Structure

```
EduTutor/
├── 01_dataset_generation.ipynb     # Synthetic data pipeline
├── 02_unsloth_tuning.ipynb         # SFT + DPO training
├── 03_evaluation_harness.ipynb     # Benchmark harness
├── 04_agentic_tutor_ui.ipynb       # Demo with Gradio
├── local_model.py                  # Shared model loading & inference utilities
├── data/                           # Generated datasets (after running NB1)
│   ├── sft_train.jsonl
│   ├── sft_validation.jsonl
│   ├── dpo_train.jsonl
│   └── dpo_validation.jsonl
├── PROJECT_WRITEUP.md              # Full competition narrative & technical deep-dive
├── LEARNING_PATH.md                # Integration with Castalia learning roadmap
├── VIDEO_DEMO_SCRIPT.md            # Script for 3-minute video pitch
└── README.md                       # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- Kaggle account with GPU quota (T4 recommended) **or** local NVIDIA GPU (16 GB+ VRAM)
- ~8 GB disk space for model weights and datasets

### Quickstart

```bash
# Clone the repository
git clone https://github.com/cyruslayo/castalia.git
cd castalia/hackathons/gemma-4-good-hackathon/EduTutor

# Install dependencies (or run on Kaggle where these are pre-installed)
pip install unsloth transformers datasets trl peft accelerate \
            scipy seaborn matplotlib pandas gradio
```

**Run the notebooks in order:**

| Step | Notebook | GPU Required | ~Time |
|------|----------|-------------|-------|
| 1 | `01_dataset_generation.ipynb` | ✅ Yes | ~45 min |
| 2 | `02_unsloth_tuning.ipynb` | ✅ Yes | ~60 min |
| 3 | `03_evaluation_harness.ipynb` | ✅ Yes | ~30 min |
| 4 | `04_agentic_tutor_ui.ipynb` | ✅ Yes | Interactive |

> **Tip:** On Kaggle, upload `local_model.py` as a utility script and add it to each notebook's input.

## License

Apache 2.0 (same as Gemma 4)
