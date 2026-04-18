# EduTutor: A Neurodiversity-Affirming AI Tutor Powered by Gemma 4

## The Problem

One in five children worldwide is neurodivergent — living with ADHD, autism, dyslexia, dyscalculia, or a combination. These children don't fail because they lack intelligence. They fail because the educational system was designed for a brain they don't have.

Current AI tutoring systems replicate the same mistakes human classrooms make: they deliver content in one format, at one pace, with one set of assumptions about how learning works. When a child with ADHD loses focus, the AI pushes forward. When an autistic student panics at an unexpected topic change, the AI ignores it. When a dyslexic child freezes at a wall of text, the AI generates more text. The result is not just academic failure — it is emotional harm.

**EduTutor exists to change this.** It is the first open-source AI tutor built from the ground up around evidence-based pedagogical strategies for neurodivergent learners.

## Our Approach

EduTutor is not a chatbot with a nicer system prompt. It is a Gemma 4 E4B model that has been fundamentally reshaped through fine-tuning and alignment to behave like an expert special education teacher — one who understands that emotional regulation comes before academic content, that productive struggle is more valuable than correct answers, and that every neurodivergent profile requires a different toolkit.

### Pedagogical Foundations

We encoded six evidence-based frameworks directly into the model's behavior through synthetic training data:

**Universal Design for Learning (UDL):** Content is offered in multiple formats. Short sentences. Bullet points. One idea at a time. The student's cognitive load is managed proactively rather than reactively.

**Zone of Proximal Development (ZPD):** The tutor follows the "I Do, We Do, You Do" scaffolding model — demonstrating first, collaborating second, then stepping back to let the student discover the answer independently. It never gives answers directly.

**Concrete-Representational-Abstract (CRA):** For math, the tutor always starts with tangible examples ("Think of pizza slices") before moving to visual representations, and only then to abstract numbers. This is critical for dyscalculic learners whose brains cannot process abstract numerals without concrete anchoring.

**Orton-Gillingham:** For literacy, the tutor follows the gold-standard structured literacy approach — explicit, sequential phonics instruction delivered through multisensory pathways.

**Zones of Regulation:** Before every response, the tutor classifies the student's emotional state into one of four zones:
- 🟢 **Green** — regulated, ready to learn
- 🟡 **Yellow** — frustrated, losing focus
- 🔴 **Red** — crisis, meltdown, wants to quit
- 🔵 **Blue** — shut down, withdrawn, "I don't care"

When a student enters Yellow, the tutor pauses academics to validate feelings. In Red, it stops all academic demands entirely and focuses only on co-regulation. In Blue, it gently re-engages with the easiest possible win.

**Spaced Repetition:** The tutor tracks concepts the student has learned and weaves past material into new lessons at increasing intervals, combating the working memory deficits common in ADHD.

### Technical Architecture

**Stage 1 — Synthetic Dataset Generation:** We generated 500+ multi-turn tutoring conversations using a teacher model, covering 4 neurodivergent profiles × 12 academic scenarios × 4 emotional zones. Each conversation demonstrates ideal pedagogical behavior. We also generated 350+ DPO preference pairs contrasting good pedagogy (scaffolding, co-regulation, short sentences) against bad pedagogy (answer-giving, ignoring emotions, text walls).

**Stage 2 — Unsloth QLoRA Fine-Tuning on Gemma 4 E4B:** We used Unsloth's optimized training pipeline to fine-tune Gemma 4 E4B with QLoRA (r=32, α=64). The two-stage training process first teaches the pedagogical persona via SFT, then aligns the model away from answer-giving behavior via DPO. The entire pipeline runs on a single T4 GPU.

**Stage 3 — Agentic Wrapper:** The fine-tuned model is wrapped in a ReAct agent loop that maintains a persistent student state machine tracking emotional zone, mastery level, difficulty, and review queue. The agent has access to tools: flashcard generation for spaced repetition, scaffolding hint retrieval, brain break suggestions, and dynamic difficulty adjustment.

**Stage 4 — Evaluation:** We built a custom LLM-as-Judge evaluation harness that scores tutor responses across five dimensions: Productive Struggle, Cognitive Load Management, Emotional Co-Regulation, Profile Adaptation, and Pedagogical Accuracy. On 10 held-out scenarios covering crisis moments, misconceptions, and all four emotional zones, EduTutor significantly outperforms base Gemma 4.

## Why Gemma 4 E4B

We chose the E4B (4 billion parameter) model deliberately. The children who need EduTutor most are the ones with the least access to cloud infrastructure — students in under-resourced schools, rural communities, and developing nations. At 4B parameters with Q4 quantization, EduTutor runs on a laptop without internet. This is not a limitation — it is the point. A tutor that requires a data center cannot help a child in a village school.

Gemma 4 E4B's 128K context window allows the tutor to maintain awareness of the full session history — critical for tracking emotional arcs and spaced repetition across a 30-minute tutoring session.

## Impact & Vision

EduTutor's vision is a world where no child is told they are "broken" because their brain works differently. The immediate impact is a free, open-source, offline-capable AI tutor that any teacher, parent, or NGO can deploy on consumer hardware to support neurodivergent students from ages 8-14.

The broader vision extends to:
- **Teacher augmentation:** EduTutor as a co-pilot that handles one-on-one scaffolding while the teacher manages 30 students
- **IEP support:** Generating data-backed insights for Individualized Education Program development
- **Global accessibility:** With Gemma 4's native support for 140+ languages, EduTutor can serve neurodivergent children regardless of their native language
- **Research platform:** The evaluation harness provides a standardized benchmark for measuring AI tutoring quality — something the field currently lacks

## Open Source Commitment

Every component is fully open:
- All 4 notebooks are public Kaggle notebooks
- The fine-tuned model weights are on Hugging Face Hub
- The synthetic dataset is available for other researchers
- The evaluation rubrics can be adapted for any AI tutoring system

EduTutor is built on Apache 2.0 Gemma 4. It stays open. Because every child deserves a tutor who understands how their brain works.
