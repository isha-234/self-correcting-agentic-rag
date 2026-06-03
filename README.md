# Multi-Agent Self-Correcting RAG

> Four independent LLM agents coordinate to answer questions over a document corpus — with self-correcting retrieval, adversarial fact-checking, and cited answers.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.14-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-teal)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![Fireworks](https://img.shields.io/badge/Embeddings-Fireworks-blue)

---

## What this project does

Most RAG systems retrieve document chunks and pass them blindly to an LLM. This system adds three things on top:

1. **Self-correcting retrieval** — a dedicated agent grades every retrieved chunk for relevance (0–1 score) and rewrites the query if quality is below threshold, retrying up to 2 times
2. **Adversarial fact-checking** — a separate Critic agent independently reviews every answer for hallucinations, missing citations, and misattributions — with no loyalty to the Answerer
3. **Confidence-scored cited answers** — every claim is traced back to a source document and page number, with an overall confidence score attached

The key difference from a multi-step agent: the Critic is a completely independent LLM instance that can override the Answerer. Two LLMs disagreeing about the same answer is multi-agent. One LLM branching is not.

---

## Architecture

```
User query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│                    LangGraph State Machine             │
│                                                        │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  Retriever   │───▶│   Answerer   │                  │
│  │   Agent      │    │    Agent     │                  │
│  │              │    │              │                  │
│  │ • Fetch top-5│    │ • Generate   │                  │
│  │   chunks     │    │   cited ans  │                  │
│  │ • Grade 0–1  │    │ • Confidence │                  │
│  │ • Rewrite if │    │   score      │                  │
│  │   score<0.6  │    └──────┬───────┘                  │
│  │ • Retry ×2   │           │                          │
│  └──────────────┘           ▼                          │
│                    ┌──────────────┐                    │
│                    │    Critic    │                    │
│                    │    Agent     │                    │
│                    │              │  approve           │
│                    │ • Fact-check │──────────▶ ┌─────┐ │
│                    │   every      │            │ Orch│ │
│                    │   claim      │  revise    │ est │ │
│                    │ • Catch      │──▶ Answerer│ rat │ │
│                    │   hallucin.  │   (once)   │ or  │ │
│                    └──────────────┘            └──┬──┘ │
│                                                   │    │
└───────────────────────────────────────────────────┼────┘
                                                    │
                                                    ▼
                                          Cited answer
                                          + confidence score
                                          + pipeline metadata
```

### Agent roster

| Agent | Model | Role | Why this model |
|-------|-------|------|----------------|
| RetrieverAgent | llama-3.1-8b-instant (Groq) | Grade chunks, rewrite queries | Fast + cheap — runs on every query, possibly twice |
| AnswererAgent | llama-3.3-70b-versatile (Groq) | Generate cited answer | Best quality — user-facing output |
| CriticAgent | llama-3.1-8b-instant (Groq) | Adversarial fact-check | Fast + cheap — verification task, not generation |
| OrchestratorAgent | llama-3.3-70b-versatile (Groq) | Synthesize final output | Full reasoning needed for edge cases |

### Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Agent orchestration | LangGraph | Handles cycles + conditional edges — the Critic→Answerer revision loop requires a graph, not a chain |
| LLM provider | Groq (OpenAI-compatible) | Free tier, 14,400 req/day, Llama 3.1/3.3 models |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 via Fireworks AI | Strong on technical text, Apache 2.0, single API key |
| Vector store | ChromaDB | Persists to disk, Python-native, zero infrastructure |
| Chunking | RecursiveCharacterTextSplitter | Splits on paragraph → sentence → word boundaries, preserves semantic context |
| API server | FastAPI | Async-native, auto-generates OpenAPI docs |
| Demo UI | Streamlit | Shows live agent trace — which agents ran, scores, verdicts |

---

## Benchmark results

Tested on 10 ML papers: Attention Is All You Need, BERT, RAG, DQN, PPO, GPT-3, LoRA, Flash Attention, Chain-of-Thought, A3C.

| Metric | Naive RAG (baseline) | Multi-Agent RAG (this project) |
|--------|---------------------|-------------------------------|
| Avg latency | ~1.0s | ~34.1s |
| Retrieval grading | ✗ none | ✅ scored per chunk (avg 0.49/1.0) |
| Avg confidence score | ✗ none | **0.81 / 1.0** |
| Re-queries triggered | 0 / 10 | 8 / 10 |
| Critic approvals | N/A | **6 / 10** |
| Critic revisions requested | N/A | 4 / 10 |
| Agents involved | 1 | 4 (Retriever, Answerer, Critic, Orchestrator) |

### Per-query breakdown

| # | Query | Naive (s) | Multi (s) | Ret. score | Critic | Revisions |
|---|-------|-----------|-----------|------------|--------|-----------|
| 1 | What is the scaled dot-product attention formula? | 1.7 | 5.1 | 0.73 | 🔄 revise | 1 |
| 2 | How many encoder and decoder layers does the base Transformer use? | 0.8 | 5.7 | 0.85 | ✅ approve | 0 |
| 3 | What are the two pre-training objectives used to train BERT? | 1.0 | 48.9 | 0.00 | 🔄 revise | 1 |
| 4 | What is the difference between BERT base and BERT large? | 1.1 | 53.7 | 0.56 | 🔄 revise | 1 |
| 5 | What retriever model does RAG use to fetch documents? | 1.0 | 35.9 | 0.55 | ✅ approve | 0 |
| 6 | How does RAG-Token differ from RAG-Sequence? | 1.3 | 37.5 | 0.52 | ✅ approve | 0 |
| 7 | What game-playing approach does DQN use to learn from raw pixels? | 0.7 | 22.7 | 0.63 | ✅ approve | 0 |
| 8 | How does DQN handle non-stationarity of the target in Q-learning? | 0.7 | 45.1 | 0.62 | 🔄 revise | 1 |
| 9 | What is the epsilon value used in PPO's clipped objective? | 0.9 | 51.0 | 0.20 | ✅ approve | 1 |
| 10 | What is the main motivation for using PPO over TRPO? | 1.1 | 34.9 | 0.20 | ✅ approve | 0 |

**Latency tradeoff is intentional** — multi-agent pipelines prioritize accuracy over speed. The naive baseline answers in ~1s but has no self-assessment, no retrieval quality check, and no fact-checking. When the system says confidence=0.00 (Q3), it genuinely doesn't know — naive RAG would hallucinate instead.

> **Q3 note:** BERT pre-training objectives scored 0.00 retrieval — the relevant section wasn't captured in the top-5 chunks. The system correctly flagged confidence=0.00 rather than hallucinating an answer. This is the intended behavior.

> **Q9 highlight:** Epsilon value scored 0.20 retrieval (low), Critic rejected the first answer, Answerer revised, Critic approved the revision. This is the full self-correction loop working as designed — low retrieval → re-query → generate → fact-check → revise → approve.

---