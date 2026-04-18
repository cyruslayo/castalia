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

EduTutor vs Base Gemma 4 E4B on 10 held-out scenarios:

| Dimension | Base Gemma 4 | EduTutor | Δ |
|---|---|---|---|
| Productive Struggle | TBD | TBD | TBD |
| Cognitive Load Management | TBD | TBD | TBD |
| Emotional Co-Regulation | TBD | TBD | TBD |
| Profile Adaptation | TBD | TBD | TBD |
| Pedagogical Accuracy | TBD | TBD | TBD |

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
├── data/                           # Generated datasets (after running NB1)
│   ├── sft_train.jsonl
│   ├── sft_validation.jsonl
│   ├── dpo_train.jsonl
│   └── dpo_validation.jsonl
└── README.md                       # This file
```

## License

Apache 2.0 (same as Gemma 4)
