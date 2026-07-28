## Benchmark - Naive RAG vs Multi-Agent RAG

| Metric | Naive RAG | Multi-Agent RAG |
|--------|-----------|----------------|
| Avg latency | 1.0s | 32.7s |
| Retrieval grading | none | scored (avg 0.51/1.0) |
| Avg confidence score | none | 0.78/1.0 |
| Re-queries triggered | 0/10 | 6/10 |
| Critic approvals | N/A | 5/10 |
| Critic revisions | N/A | 5/10 |
| Agents | 1 | 3 (Retriever, Answerer, Critic) |

### Per-query breakdown

| # | Query | Naive (s) | Multi (s) | Ret. score | Critic | Revisions |
|---|-------|-----------|-----------|------------|--------|-----------|
| 1 | What is the scaled dot-product attention f... | 1.2 | 3.3 | 0.70 | revise | 1 |
| 2 | How many encoder and decoder layers does t... | 0.5 | 1.7 | 0.73 | approve | 0 |
| 3 | What are the two pre-training objectives u... | 1.0 | 40.4 | 0.48 | revise | 1 |
| 4 | What is the difference between BERT base a... | 1.0 | 30.5 | 0.63 | approve | 0 |
| 5 | What retriever model does RAG use to fetch... | 0.8 | 47.9 | 0.56 | approve | 1 |
| 6 | How does RAG-Token differ from RAG-Sequenc... | 1.7 | 37.3 | 0.42 | approve | 0 |
| 7 | What game-playing approach does DQN use to... | 0.9 | 22.1 | 0.63 | approve | 0 |
| 8 | How does DQN handle the non-stationarity o... | 0.8 | 48.6 | 0.35 | revise | 1 |
| 9 | What is the epsilon value used in PPO's cl... | 0.8 | 50.4 | 0.32 | revise | 1 |
| 10 | What is the main motivation for using PPO ... | 0.9 | 44.9 | 0.24 | revise | 1 |

### Key takeaways

- Self-correcting retrieval: 6/10 queries triggered a re-query when initial retrieval scored below threshold
- Independent fact-checking: Critic reviewed all 10 answers, requesting revisions for 5
- Confidence scoring: Multi-agent pipeline produces calibrated confidence (avg 0.78) - naive RAG has no self-assessment

## Comparative analysis — CriticAgent vs RAGAS (independent judge)

RAGAS is a separate evaluation library — it grades each answer using its
own judge LLM, independent of your CriticAgent. A mismatch (Critic approved,
but RAGAS faithfulness is low) is worth manually reviewing.

Note: this uses RAGAS's reference-free faithfulness metric only (no
hand-written ground truth answers are used), so treat it as a second
opinion rather than an objective ground truth.

| # | Query | Critic verdict | RAGAS faithfulness | Agreement |
|---|-------|-----------------|---------------------|------------|
| 1 | What is the scaled dot-product attention... | revise | 1.00 | mismatch |
| 2 | How many encoder and decoder layers does... | approve | 1.00 | agree |
| 3 | What are the two pre-training objectives... | revise | 0.00 | agree |
| 4 | What is the difference between BERT base... | approve | 1.00 | agree |
| 5 | What retriever model does RAG use to fet... | approve | 1.00 | agree |
| 6 | How does RAG-Token differ from RAG-Seque... | approve | 1.00 | agree |
| 7 | What game-playing approach does DQN use ... | approve | 1.00 | agree |
| 8 | How does DQN handle the non-stationarity... | revise | 0.00 | agree |
| 9 | What is the epsilon value used in PPO's ... | revise | 1.00 | mismatch |
| 10 | What is the main motivation for using PP... | revise | 1.00 | mismatch |

3/10 queries showed disagreement between the Critic and RAGAS — worth reviewing individually.
