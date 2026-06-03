"""
eval.py — Benchmark: Naive RAG vs Multi-Agent RAG
===================================================
Runs 10 queries through both pipelines and builds the README table.

Usage:
    python eval.py
    python eval.py --collection my_rag --queries_file my_queries.txt
"""

import os
import json
import time
import argparse
import statistics
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Fireworks client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL_ANSWERER = os.getenv("MODEL_ANSWERER")

# Default benchmark queries — replace with queries relevant to YOUR documents
DEFAULT_QUERIES = [
    # Transformer (1706.03762) — specific, answerable
    "What is the scaled dot-product attention formula?",
    "How many encoder and decoder layers does the base Transformer model use?",

    # BERT (1810.04805) — specific, answerable
    "What are the two pre-training objectives used to train BERT?",
    "What is the difference between BERT base and BERT large?",

    # RAG (2005.11401) — specific, answerable
    "What retriever model does RAG use to fetch documents?",
    "How does RAG-Token differ from RAG-Sequence?",

    # DQN (1312.5602) — specific, answerable
    "What game-playing approach does DQN use to learn from raw pixels?",
    "How does DQN handle the non-stationarity of the target in Q-learning?",

    # PPO (1707.06347) — specific, answerable
    "What is the epsilon value used in PPO's clipped objective?",
    "What is the main motivation for using PPO over TRPO?",
]


# ─── Naive RAG baseline ───────────────────────────────────────────────────────

class NaiveRAG:
    """Single LLM call — no grading, no re-query, no critic."""

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> dict:
        t0 = time.perf_counter()
        context_string, _ = self.retriever.retrieve_as_context_string(query)

        response = client.chat.completions.create(
            model=MODEL_ANSWERER,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided context. "
                               "Cite sources with [N] notation.",
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context_string}",
                },
            ],
        )
        return {
            "answer":             response.choices[0].message.content,
            "latency":            round(time.perf_counter() - t0, 3),
            "retrieval_attempts": 1,
            "revision_count":     0,
            "confidence":         None,
        }


# ─── Eval runner ─────────────────────────────────────────────────────────────

def run_eval(collection_name: str, queries: list) -> dict:
    from retriever import Retriever
    from rag_graph import MultiAgentRAGPipeline

    print("\n  Benchmark: Naive RAG vs Multi-Agent RAG (Groq + Fireworks)")
    print("=" * 60)

    retriever  = Retriever(collection_name=collection_name)
    naive_rag  = NaiveRAG(retriever=retriever)
    multi_rag  = MultiAgentRAGPipeline(collection_name=collection_name)

    naive_results, multi_results = [], []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query[:55]}...")

        print("  Naive RAG...")
        naive = naive_rag.run(query)
        naive_results.append({**naive, "query": query})

        print("  Multi-Agent RAG...")
        state  = multi_rag.run(query, verbose=False)
        final  = state.get("final_result", {})
        retr   = state.get("retriever_result", {})
        critic = state.get("critic_result", {})
        multi_results.append({
            "query":              query,
            "answer":             final.get("final_answer", ""),
            "latency":            state["node_timings"].get("total", 0),
            "retrieval_attempts": retr.get("attempts", 1),
            "retrieval_score":    retr.get("avg_score", 0),
            "critic_verdict":     critic.get("verdict", "unknown"),
            "revision_count":     state.get("revision_count", 0),
            "confidence":         final.get("confidence", 0),
        })
        print(f"  Naive: {naive['latency']:.1f}s | Multi: {multi_results[-1]['latency']:.1f}s")

    return {"naive": naive_results, "multi": multi_results}


# ─── Table builder ────────────────────────────────────────────────────────────

def build_table(results: dict) -> str:
    naive = results["naive"]
    multi = results["multi"]

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return statistics.mean(vals) if vals else 0

    n_lat   = avg(naive, "latency")
    m_lat   = avg(multi, "latency")
    m_score = avg(multi, "retrieval_score")
    m_conf  = avg(multi, "confidence")
    requery = sum(1 for r in multi if r["retrieval_attempts"] > 1)
    revised = sum(1 for r in multi if r["critic_verdict"] == "revise")
    approved = len(multi) - revised

    table = f"""## Benchmark — Naive RAG vs Multi-Agent RAG (Fireworks AI)

| Metric | Naive RAG | Multi-Agent RAG |
|--------|-----------|----------------|
| Avg latency | {n_lat:.1f}s | {m_lat:.1f}s |
| Retrieval grading | ✗ none | ✅ scored (avg {m_score:.2f}/1.0) |
| Avg confidence score | ✗ none | {m_conf:.2f}/1.0 |
| Re-queries triggered | 0/{len(naive)} | {requery}/{len(multi)} |
| Critic approvals | N/A | {approved}/{len(multi)} |
| Critic revisions | N/A | {revised}/{len(multi)} |
| Agents | 1 | 4 (Retriever, Answerer, Critic, Orchestrator) |

### Per-query breakdown

| # | Query | Naive (s) | Multi (s) | Ret. score | Critic | Revisions |
|---|-------|-----------|-----------|------------|--------|-----------|
"""
    for i, (n, m) in enumerate(zip(naive, multi), 1):
        q  = (n["query"][:42] + "...") if len(n["query"]) > 42 else n["query"]
        ic = "✅" if m["critic_verdict"] == "approve" else "🔄"
        table += (f"| {i} | {q} | {n['latency']:.1f} | {m['latency']:.1f} | "
                  f"{m['retrieval_score']:.2f} | {ic} {m['critic_verdict']} | "
                  f"{m['revision_count']} |\n")

    table += f"""
### Key takeaways

- **Self-correcting retrieval**: {requery}/{len(multi)} queries triggered a re-query when initial retrieval scored below threshold
- **Independent fact-checking**: Critic reviewed all {len(multi)} answers, requesting revisions for {revised}
- **Confidence scoring**: Multi-agent pipeline produces calibrated confidence (avg {m_conf:.2f}) — naive RAG has no self-assessment
- **Provider**: Groq (llama-3.1-8b-instant for Retriever+Critic, llama-3.3-70b-versatile for Answerer+Orchestrator) + Fireworks AI (nomic-embed-text-v1.5 for embeddings)
"""
    return table


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection",   default=os.getenv("CHROMA_COLLECTION", "rag_collection"))
    parser.add_argument("--queries_file", default=None, help="Path to .txt file, one query per line")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file and Path(args.queries_file).exists():
        queries = [l.strip() for l in Path(args.queries_file).read_text().splitlines() if l.strip()]
        print(f"📋  Loaded {len(queries)} queries from {args.queries_file}")

    results = run_eval(args.collection, queries)

    Path("eval_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n📄  Raw results → eval_results.json")

    table = build_table(results)
    Path("README_table.md").write_text(table)
    print(f"📊  README table → README_table.md\n")
    print(table)


if __name__ == "__main__":
    main()
