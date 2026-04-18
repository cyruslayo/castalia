# EduTutor — Video Demo Script

**Target length:** 3-5 minutes  
**Tone:** Warm, passionate, human — this is about children, not technology  
**Format:** Screen recording + voiceover with face cam overlay

---

## Scene 1: The Hook (0:00 - 0:30)

**[Screen: Black background with text appearing word by word]**

> "One in five children is neurodivergent."
> "They don't need a slower version of the same teaching."
> "They need a fundamentally different approach."

**[Cut to: Face cam]**

**VOICEOVER:** "Hi, I'm [Name]. This is EduTutor — an AI tutor that actually understands how neurodivergent children learn. Not because it was told to be 'patient' in a system prompt. Because it was trained on evidence-based special education strategies from the ground up."

---

## Scene 2: The Problem (0:30 - 1:15)

**[Screen: Split-screen comparison]**

**VOICEOVER:** "Let me show you what happens when a frustrated child with ADHD talks to a normal AI versus EduTutor."

**[Left side: Base Gemma 4 response to "I HATE FRACTIONS! I'm so STUPID!"]**
- Show the base model giving a long explanation of fractions
- Highlight: ignores emotions, gives the answer, wall of text

**[Right side: EduTutor response to the same message]**
- Show EduTutor: validates feelings first, offers a brain break, then scaffolds with pizza analogy
- Highlight: short sentences, emotional awareness, no answer given

**VOICEOVER:** "The base model gives a correct math explanation. But that child isn't listening — they're in crisis. EduTutor knows that emotional regulation comes first. It uses the Zones of Regulation framework to detect that this child is in the RED zone, and it responds accordingly."

---

## Scene 3: Live Demo (1:15 - 3:00)

**[Screen: Gradio UI running in the notebook]**

**VOICEOVER:** "Let me walk you through a real session."

### Demo Flow:

1. **Set profile to ADHD, subject to Math, topic to Fractions**
   - "We start by selecting the learner's profile. This changes the underlying strategies."

2. **Student (GREEN): "Hi! Can you help me with fractions?"**
   - Show EduTutor's warm greeting, setting expectations
   - Point out the dashboard: Zone = GREEN

3. **Student (GREEN): "So 1/4 + 2/4 = 3/8, right?"**
   - Show EduTutor NOT saying "wrong" — instead using the pizza analogy
   - "Watch — it doesn't say 'that's wrong.' It validates the attempt and uses a concrete example."

4. **Student (YELLOW): "Ugh, this is confusing. Can we just skip it?"**
   - Show dashboard change: Zone = YELLOW
   - Show EduTutor pausing academics, validating frustration
   - "The zone classifier detected rising frustration. The tutor immediately shifts from teaching to support."

5. **Student (RED): "I CAN'T DO THIS! I'm the WORST!"**
   - Show dashboard: Zone = RED, difficulty auto-decreased
   - Show EduTutor's co-regulation response: breathing exercise, no academics
   - "In a RED zone, all academic demands stop. The only job is helping the child feel safe."

6. **Student (GREEN again): "Okay... I'll try again. Slowly."**
   - Show EduTutor returning to the problem at a lower difficulty
   - "And when they're ready, we resume — but at a gentler level."

**VOICEOVER:** "This isn't just a chatbot. It's an agent with a state machine tracking emotional zones, mastery, and difficulty across the entire session."

---

## Scene 4: Technical Depth (3:00 - 3:45)

**[Screen: Architecture diagram from README]**

**VOICEOVER:** "Under the hood, EduTutor uses a 3-stage pipeline:"

1. "We synthesized 500+ tutoring conversations across 4 neurodivergent profiles, 12 academic scenarios, and all 4 emotional zones."

2. "We fine-tuned Gemma 4 E4B using Unsloth QLoRA — first SFT to teach the persona, then DPO alignment to prevent answer-giving."

3. "We wrapped the model in a ReAct agent with tools for flashcards, scaffolding hints, and brain breaks."

**[Screen: Evaluation results chart from Notebook 3]**

**VOICEOVER:** "Our LLM-as-Judge evaluation shows EduTutor outperforms base Gemma 4 on every dimension — especially emotional co-regulation and productive struggle."

---

## Scene 5: Why It Matters (3:45 - 4:15)

**[Screen: Face cam, slow zoom]**

**VOICEOVER:** "At 4 billion parameters, quantized to GGUF, EduTutor runs on a laptop — no cloud, no internet. The children who need this most are in under-resourced schools where bandwidth is a luxury. Gemma 4 E4B makes that possible."

"Everything is open source. The model, the data, the evaluation framework. Because every child deserves a tutor who understands how their brain works."

---

## Scene 6: Close (4:15 - 4:30)

**[Screen: EduTutor logo + links]**

**VOICEOVER:** "EduTutor. Built with Gemma 4, fine-tuned with Unsloth, designed for every kind of mind."

**[Show links:]**
- Kaggle notebooks
- GitHub repository
- Hugging Face model

**[End card]**
