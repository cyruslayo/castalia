# Castalia Labs — Applied AI Product Engineering

> **Post-Curriculum Track** · Assumes Orbits 0–6 complete · 12 labs × 4 notebooks = 48 notebooks · All run on Colab Pro T4

## What are the labs?

The Castalia curriculum teaches you *how AI works*. The labs teach you *how to build products with it*.

Each lab targets a **proven product category** where companies already generate $50M–$1B+ in annual recurring revenue. You start from first principles — understanding *why* the problem exists and *what fundamentally changed* when language models arrived — then build a working prototype that could become a real product.

Every lab follows the same arc:

| Notebook | Purpose |
|----------|---------|
| `00_first_principles.ipynb` | Derive the problem: what it is, why it's hard, what pre-LLM solutions existed, why they failed, what LLMs change |
| `01_architecture.ipynb` | Design the system: components, data flow, API contracts, technology selection |
| `02_build.ipynb` | Build it: full working implementation combining techniques from multiple Castalia domains |
| `03_evaluate.ipynb` | Harden it: evaluation harness, failure analysis, edge cases, production readiness |

---

## Prerequisites

Before starting any lab, you should have completed:

- **Orbit 0 — Foundations**: Python, ML basics, transformer internals
- **Orbit 1 — Prompt Engineering**: Structured prompting, chain-of-thought, few-shot
- **Orbit 2 — RAG**: Embeddings, vector stores, retrieval strategies, hybrid search
- **Orbit 3 — Agents**: Tool use, planning, ReAct, multi-agent systems
- **Orbit 4 — Evals**: Metrics, benchmarks, LLM-as-judge, regression testing
- **Orbit 5 — Finetuning**: LoRA, data preparation, training loops, merging
- **Orbit 6 — Multimodal / Systems**: Vision-language models, serving, deployment

---

## The 12 labs

| # | Lab | What you build | Market | Key domains |
|---|-----|----------------|--------|-------------|
| 01 | [Customer Support Agent](01_customer_support_agent/) | Knowledge-grounded support bot with confidence-based escalation | $12B+ | RAG · Agents · Evals · PE |
| 02 | [Sales Intelligence & Outreach](02_sales_intelligence/) | AI SDR with prospect research and personalized outreach | $5B+ | Agents · RAG · PE · Evals |
| 03 | [Contract Analyzer](03_contract_analyzer/) | Clause-level risk scoring with negotiation suggestions | $28B+ | Multimodal · RAG · Agents · Evals |
| 04 | [Compliance Auditor](04_compliance_auditor/) | Semantic regulation-to-policy matching with gap analysis | $35B+ | RAG · Agents · Evals · PE |
| 05 | [Document Processing (IDP)](05_document_processing/) | VLM-powered document understanding and field extraction | $12B+ | Multimodal · Agents · RAG · Evals |
| 06 | [AI Data Analyst](06_ai_data_analyst/) | Text-to-SQL with self-correction and insight narration | $30B+ | Agents · RAG · Evals · PE |
| 07 | [Content & SEO Engine](07_content_seo_engine/) | Research-driven content generation with brand voice | $400B+ | RAG · Agents · Evals · PE |
| 08 | [Recruitment Matching](08_recruitment_matching/) | Semantic candidate-job matching with bias detection | $200B+ | RAG · Agents · Evals · Multimodal |
| 09 | [Feedback Intelligence](09_feedback_intelligence/) | Aspect-level opinion mining with trend detection | $12B+ | RAG · Agents · Evals · Finetuning |
| 10 | [Code Review & Security](10_code_review_security/) | Data-flow-aware vulnerability detection and fix generation | $10B+ | RAG · Agents · Evals · PE |
| 11 | [Proposal & RFP Automation](11_proposal_rfp_automation/) | Context-aware answer retrieval with cross-section consistency | $3B+ | RAG · Agents · Evals · Multimodal |
| 12 | [Enterprise Search & Q&A](12_enterprise_search/) | Multi-source knowledge Q&A with authority ranking | $8B+ | RAG · Agents · Evals · Systems |

---

## Domain coverage matrix

Every lab uses **Prompt Engineering + RAG + Agents + Evals** as a baseline. Additional domains provide depth:

| Lab | PE | RAG | Agents | Evals | Finetuning | Multimodal | Systems |
|-----|:--:|:---:|:------:|:-----:|:----------:|:----------:|:-------:|
| 01 Customer Support | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| 02 Sales Intelligence | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| 03 Contract Analyzer | ✓ | ✓ | ✓ | ✓ | | ✓ | |
| 04 Compliance Auditor | ✓ | ✓ | ✓ | ✓ | | ✓ | |
| 05 Document Processing | ✓ | ✓ | ✓ | ✓ | | ✓ | |
| 06 AI Data Analyst | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| 07 Content & SEO | ✓ | ✓ | ✓ | ✓ | | | |
| 08 Recruitment Matching | ✓ | ✓ | ✓ | ✓ | | ✓ | |
| 09 Feedback Intelligence | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| 10 Code Review & Security | ✓ | ✓ | ✓ | ✓ | | | |
| 11 Proposal/RFP Automation | ✓ | ✓ | ✓ | ✓ | | ✓ | |
| 12 Enterprise Search & Q&A | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |

---

## Suggested progression

Labs are designed to be independent — pick any one that interests you. But if you want a guided path:

1. **Start here**: Lab 01 (Customer Support) — the most accessible pipeline pattern with immediate results
2. **Build confidence**: Labs 06, 09, 07 — clear problem spaces, strong eval stories
3. **Go deeper**: Labs 03, 04, 05 — document-heavy, require multimodal + structured reasoning
4. **Cross-domain mastery**: Labs 02, 08, 10, 11 — multi-agent systems, specialized domains
5. **Capstone**: Lab 12 (Enterprise Search) — the most complex, requires systems thinking + access control

---

## ROI tier ranking

If your goal is to turn a lab into a real product, here's where the market opportunity is strongest:

| Tier | Labs | Why |
|------|------|-----|
| **Tier 1** — Highest ROI | 04 (Compliance), 12 (Enterprise Search), 01 (Customer Support) | Urgent pain, clear pricing, massive markets |
| **Tier 2** — Strong ROI | 03 (Contracts), 05 (IDP), 02 (Sales) | Direct cost arbitrage, proven demand |
| **Tier 3** — Large markets | 06 (Data Analyst), 09 (Feedback), 08 (Recruiting) | Big TAM but more competition |
| **Tier 4** — Solid opportunities | 10 (Code Review), 11 (RFP), 07 (Content) | Proven markets, differentiation needed |

---

## How to use these labs

1. **Read the first principles notebook first** — don't skip it. Understanding *why* the problem is hard is what separates engineers who build products from engineers who build demos
2. **Run every cell** — the architecture notebook has working component prototypes, not just diagrams
3. **Do the exercises** — each notebook has exercises that extend the core implementation
4. **Read the eval notebook even if you don't run it** — understanding *how to measure quality* is the most underrated product skill
5. **Check the "Path to product" section** — every lab includes concrete next steps for turning the prototype into an MVP

---

## Technical requirements

- **Runtime**: Google Colab Pro with T4 GPU
- **API keys**: OpenAI API key (required for all labs). Some labs optionally use Anthropic, Google, or other APIs
- **Storage**: Google Drive for model caching (`/content/drive/MyDrive/models`)
- **Python**: 3.10+ (Colab default)

---

## References

- [a16z Top 100 Gen AI Consumer Apps](https://a16z.com/100-gen-ai-apps/) — market validation for AI product categories
- [Sequoia AI 50](https://www.sequoiacap.com/article/ai-50-2024/) — top AI companies by growth and market fit
- Castalia curriculum: [foundations/](../foundations/) · [prompt-engineering/](../prompt-engineering/) · [rag/](../rag/) · [agents/](../agents/) · [evals/](../evals/) · [finetuning/](../finetuning/) · [multimodal/](../multimodal/) · [systems/](../systems/)
