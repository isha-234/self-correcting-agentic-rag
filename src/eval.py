"""
eval.py — Benchmark: Naive RAG vs Multi-Agent RAG (+ RAGAS comparison)
========================================================================
Runs queries through both pipelines, builds the README table, and
optionally runs RAGAS as an independent check on the Multi-Agent
pipeline's answers — comparing RAGAS's faithfulness score against your
own Critic's verdicts to see where they agree or disagree.

Usage:
    python eval.py
    python eval.py --collection my_rag --queries_file my_queries.txt
    python eval.py --skip_ragas
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

# LLM client (OpenAI-compatible)
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
MODEL_ANSWERER = os.getenv("MODEL_ANSWERER")

# Model used as the RAGAS judge — defaults to the same model as the Answerer
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", MODEL_ANSWERER)

# Default benchmark queries - replace with queries relevant to YOUR documents
DEFAULT_QUERIES = [
    # Transformer (1706.03762) - specific, answerable
    "What is the scaled dot-product attention formula?",
    "How many encoder and decoder layers does the base Transformer model use?",

    # BERT (1810.04805) - specific, answerable
    "What are the two pre-training objectives used to train BERT?",
    "What is the difference between BERT base and BERT large?",

    # RAG (2005.11401) - specific, answerable
    "What retriever model does RAG use to fetch documents?",
    "How does RAG-Token differ from RAG-Sequence?",

    # DQN (1312.5602) - specific, answerable
    "What game-playing approach does DQN use to learn from raw pixels?",
    "How does DQN handle the non-stationarity of the target in Q-learning?",

    # PPO (1707.06347) - specific, answerable
    "What is the epsilon value used in PPO's clipped objective?",
    "What is the main motivation for using PPO over TRPO?",
]


# --- Naive RAG baseline --------------------------------------------------------

class NaiveRAG:
    """Single LLM call - no grading, no re-query, no critic."""

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


# --- Eval runner ----------------------------------------------------------------

def run_eval(collection_name: str, queries: list) -> dict:
    from retriever import Retriever
    from rag_graph import MultiAgentRAGPipeline

    print("\n  Benchmark: Naive RAG vs Multi-Agent RAG")
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
            # keep the raw retrieved chunk texts too — RAGAS needs the
            # actual context strings, not just the retrieval score
            "contexts":           [c.text for c in retr.get("chunks", [])],
            "latency":            state["node_timings"].get("total", 0),
            "retrieval_attempts": retr.get("attempts", 1),
            "retrieval_score":    retr.get("avg_score", 0),
            "critic_verdict":     critic.get("verdict", "unknown"),
            "revision_count":     state.get("revision_count", 0),
            "confidence":         final.get("confidence", 0),
        })
        print(f"  Naive: {naive['latency']:.1f}s | Multi: {multi_results[-1]['latency']:.1f}s")

    return {"naive": naive_results, "multi": multi_results}


# --- RAGAS comparison ------------------------------------------------------------
# Runs RAGAS as an independent judge on the Multi-Agent pipeline's answers.
# This is a second, separate opinion from your own CriticAgent — useful for
# checking whether your Critic's approvals are actually well grounded.
#
# Only faithfulness is used here since it doesn't require a hand-written
# ground truth ("reference") answer for each query. Context precision was
# dropped because current RAGAS versions require a reference column for it,
# which this project doesn't maintain.

def run_ragas_eval(multi_results: list) -> list:
    """
    Returns a list of dicts, one per query, each with a RAGAS faithfulness
    score. Falls back to None scores if RAGAS is not installed or a query
    fails, rather than stopping the whole run.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("\nRAGAS (or its dependencies) is not installed - skipping RAGAS comparison.")
        print("Install with: pip install ragas datasets langchain-openai")
        return [{"faithfulness": None} for _ in multi_results]

    judge_llm = ChatOpenAI(
        model=RAGAS_JUDGE_MODEL,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    )

    dataset = Dataset.from_list([
        {
            "question": r["query"],
            "answer":   r["answer"],
            "contexts": r["contexts"] if r["contexts"] else [""],
        }
        for r in multi_results
    ])

    print("\nRunning RAGAS evaluation (independent judge)...")
    scored = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=judge_llm,
    )
    scored_df = scored.to_pandas()

    return [
        {
            "faithfulness": round(float(row["faithfulness"]), 3) if row["faithfulness"] == row["faithfulness"] else None,
        }
        for _, row in scored_df.iterrows()
    ]


def build_ragas_comparison_table(multi_results: list, ragas_scores: list) -> str:
    """
    Puts your Critic's verdict side by side with RAGAS's independent
    faithfulness score, so disagreements are easy to spot.
    """
    table = """## Comparative analysis — CriticAgent vs RAGAS (independent judge)

RAGAS is a separate evaluation library — it grades each answer using its
own judge LLM, independent of your CriticAgent. A mismatch (Critic approved,
but RAGAS faithfulness is low) is worth manually reviewing.

Note: this uses RAGAS's reference-free faithfulness metric only (no
hand-written ground truth answers are used), so treat it as a second
opinion rather than an objective ground truth.

| # | Query | Critic verdict | RAGAS faithfulness | Agreement |
|---|-------|-----------------|---------------------|------------|
"""
    mismatches = 0
    for i, (m, r) in enumerate(zip(multi_results, ragas_scores), 1):
        q = (m["query"][:40] + "...") if len(m["query"]) > 40 else m["query"]
        faith = r["faithfulness"]
        verdict = m["critic_verdict"]

        if faith is None:
            agreement = "n/a"
        elif verdict == "approve" and faith >= 0.7:
            agreement = "agree"
        elif verdict == "revise" and faith < 0.7:
            agreement = "agree"
        else:
            agreement = "mismatch"
            mismatches += 1

        faith_str = f"{faith:.2f}" if faith is not None else "n/a"
        table += f"| {i} | {q} | {verdict} | {faith_str} | {agreement} |\n"

    table += f"\n{mismatches}/{len(multi_results)} queries showed disagreement between the Critic and RAGAS — worth reviewing individually.\n"
    return table


# --- Table builder ---------------------------------------------------------------

def build_table(results: dict) -> str:
    naive = results["naive"]
    multi = results["multi"]

    def avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return statistics.mean(vals) if vals else 0

    n_lat    = avg(naive, "latency")
    m_lat    = avg(multi, "latency")
    m_score  = avg(multi, "retrieval_score")
    m_conf   = avg(multi, "confidence")
    requery  = sum(1 for r in multi if r["retrieval_attempts"] > 1)
    revised  = sum(1 for r in multi if r["critic_verdict"] == "revise")
    approved = len(multi) - revised

    table = f"""## Benchmark - Naive RAG vs Multi-Agent RAG

| Metric | Naive RAG | Multi-Agent RAG |
|--------|-----------|----------------|
| Avg latency | {n_lat:.1f}s | {m_lat:.1f}s |
| Retrieval grading | none | scored (avg {m_score:.2f}/1.0) |
| Avg confidence score | none | {m_conf:.2f}/1.0 |
| Re-queries triggered | 0/{len(naive)} | {requery}/{len(multi)} |
| Critic approvals | N/A | {approved}/{len(multi)} |
| Critic revisions | N/A | {revised}/{len(multi)} |
| Agents | 1 | 3 (Retriever, Answerer, Critic) |

### Per-query breakdown

| # | Query | Naive (s) | Multi (s) | Ret. score | Critic | Revisions |
|---|-------|-----------|-----------|------------|--------|-----------|
"""
    for i, (n, m) in enumerate(zip(naive, multi), 1):
        q  = (n["query"][:42] + "...") if len(n["query"]) > 42 else n["query"]
        table += (f"| {i} | {q} | {n['latency']:.1f} | {m['latency']:.1f} | "
                  f"{m['retrieval_score']:.2f} | {m['critic_verdict']} | "
                  f"{m['revision_count']} |\n")

    table += f"""
### Key takeaways

- Self-correcting retrieval: {requery}/{len(multi)} queries triggered a re-query when initial retrieval scored below threshold
- Independent fact-checking: Critic reviewed all {len(multi)} answers, requesting revisions for {revised}
- Confidence scoring: Multi-agent pipeline produces calibrated confidence (avg {m_conf:.2f}) - naive RAG has no self-assessment
"""
    return table


# --- CLI -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection",   default=os.getenv("CHROMA_COLLECTION", "rag_collection"))
    parser.add_argument("--queries_file", default=None, help="Path to .txt file, one query per line")
    parser.add_argument("--skip_ragas",   action="store_true", help="Skip the RAGAS comparison pass")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file and Path(args.queries_file).exists():
        queries = [l.strip() for l in Path(args.queries_file).read_text().splitlines() if l.strip()]
        print(f"Loaded {len(queries)} queries from {args.queries_file}")

    results = run_eval(args.collection, queries)

    Path("eval_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nRaw results written to eval_results.json")

    table = build_table(results)

    if not args.skip_ragas:
        ragas_scores = run_ragas_eval(results["multi"])
        table += "\n" + build_ragas_comparison_table(results["multi"], ragas_scores)

    Path("README_table.md").write_text(table)
    print(f"README table written to README_table.md\n")
    print(table)


if __name__ == "__main__":
    main()