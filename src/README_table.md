## Benchmark — Naive RAG vs Multi-Agent RAG (Fireworks AI)

| Metric | Naive RAG | Multi-Agent RAG |
|--------|-----------|----------------|
| Avg latency | 1.0s | 34.1s |
| Retrieval grading | ✗ none | ✅ scored (avg 0.49/1.0) |
| Avg confidence score | ✗ none | 0.81/1.0 |
| Re-queries triggered | 0/10 | 8/10 |
| Critic approvals | N/A | 6/10 |
| Critic revisions | N/A | 4/10 |
| Agents | 1 | 4 (Retriever, Answerer, Critic, Orchestrator) |

### Per-query breakdown

| # | Query | Naive (s) | Multi (s) | Ret. score | Critic | Revisions |
|---|-------|-----------|-----------|------------|--------|-----------|
| 1 | What is the scaled dot-product attention f... | 1.7 | 5.1 | 0.73 | 🔄 revise | 1 |
| 2 | How many encoder and decoder layers does t... | 0.8 | 5.7 | 0.85 | ✅ approve | 0 |
| 3 | What are the two pre-training objectives u... | 1.0 | 48.9 | 0.00 | 🔄 revise | 1 |
| 4 | What is the difference between BERT base a... | 1.1 | 53.7 | 0.56 | 🔄 revise | 1 |
| 5 | What retriever model does RAG use to fetch... | 1.0 | 35.9 | 0.55 | ✅ approve | 0 |
| 6 | How does RAG-Token differ from RAG-Sequenc... | 1.3 | 37.5 | 0.52 | ✅ approve | 0 |
| 7 | What game-playing approach does DQN use to... | 0.7 | 22.7 | 0.63 | ✅ approve | 0 |
| 8 | How does DQN handle the non-stationarity o... | 0.7 | 45.1 | 0.62 | 🔄 revise | 1 |
| 9 | What is the epsilon value used in PPO's cl... | 0.9 | 51.0 | 0.20 | ✅ approve | 1 |
| 10 | What is the main motivation for using PPO ... | 1.1 | 34.9 | 0.20 | ✅ approve | 0 |

### Key takeaways

- **Self-correcting retrieval**: 8/10 queries triggered a re-query when initial retrieval scored below threshold
- **Independent fact-checking**: Critic reviewed all 10 answers, requesting revisions for 4
- **Confidence scoring**: Multi-agent pipeline produces calibrated confidence (avg 0.81) — naive RAG has no self-assessment
- **Provider**: Groq (llama-3.1-8b-instant for Retriever+Critic, llama-3.3-70b-versatile for Answerer+Orchestrator) + Fireworks AI (nomic-embed-text-v1.5 for embeddings)
