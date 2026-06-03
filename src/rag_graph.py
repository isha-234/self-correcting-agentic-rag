"""
rag_graph.py — LangGraph Multi-Agent Pipeline
===============================================
Wires the 4 agents into a directed state machine.

Flow:
  retrieve → answer → critique
                         ↓ approve → orchestrate → END
                         ↓ revise  → answer (once) → orchestrate → END
"""

import os
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from retriever import Retriever
from agents import RetrieverAgent, AnswererAgent, CriticAgent, OrchestratorAgent

from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

MAX_REVISIONS = int(os.getenv("MAX_CRITIC_REVISIONS", "1"))


# ─── Shared State ─────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query:            str
    retriever_result: Optional[dict]
    answerer_result:  Optional[dict]
    critic_result:    Optional[dict]
    final_result:     Optional[dict]
    revision_count:   int
    start_time:       float
    node_timings:     dict


# ─── Node Functions ───────────────────────────────────────────────────────────

def run_retriever(state: RAGState, agent: RetrieverAgent) -> dict:
    t0     = time.perf_counter()
    print(f"\n[Graph] → RetrieverAgent")
    result = agent.run(state["query"])
    timings = {**state.get("node_timings", {}), "retriever": round(time.perf_counter() - t0, 3)}
    return {"retriever_result": result, "node_timings": timings}


def run_answerer(state: RAGState, agent: AnswererAgent) -> dict:
    t0 = time.perf_counter()
    print(f"\n[Graph] → AnswererAgent (revision #{state['revision_count']})")
    result  = agent.run(
        query=state["query"],
        context_string=state["retriever_result"]["context_string"],
    )
    key     = f"answerer_r{state['revision_count']}"
    timings = {**state.get("node_timings", {}), key: round(time.perf_counter() - t0, 3)}
    return {"answerer_result": result, "node_timings": timings}


def run_critic(state: RAGState, agent: CriticAgent) -> dict:
    t0 = time.perf_counter()
    print(f"\n[Graph] → CriticAgent")
    result  = agent.run(
        query=state["query"],
        context_string=state["retriever_result"]["context_string"],
        draft_answer=state["answerer_result"].get("answer", ""),
    )
    timings = {**state.get("node_timings", {}), "critic": round(time.perf_counter() - t0, 3)}
    return {"critic_result": result, "node_timings": timings}


def run_orchestrator(state: RAGState, agent: OrchestratorAgent) -> dict:
    t0 = time.perf_counter()
    print(f"\n[Graph] → OrchestratorAgent")
    result  = agent.run(
        query=state["query"],
        retriever_result=state["retriever_result"],
        answerer_result=state["answerer_result"],
        critic_result=state["critic_result"],
    )
    total   = time.perf_counter() - state["start_time"]
    timings = {
        **state.get("node_timings", {}),
        "orchestrator": round(time.perf_counter() - t0, 3),
        "total": round(total, 3),
    }
    return {"final_result": result, "node_timings": timings}


def increment_revision(state: RAGState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


# ─── Conditional Routing ──────────────────────────────────────────────────────

def route_after_critic(state: RAGState) -> str:
    verdict        = state.get("critic_result", {}).get("verdict", "approve")
    revision_count = state.get("revision_count", 0)

    if verdict == "approve":
        print(f"\n[Graph] Critic approved → Orchestrator")
        return "orchestrate"
    if revision_count >= MAX_REVISIONS:
        print(f"\n[Graph] Max revisions reached → Orchestrator")
        return "orchestrate"

    print(f"\n[Graph] Critic rejected → Answerer revision #{revision_count + 1}")
    return "revise"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_graph(
    retriever_agent:    RetrieverAgent,
    answerer_agent:     AnswererAgent,
    critic_agent:       CriticAgent,
    orchestrator_agent: OrchestratorAgent,
):
    def retriever_node(state):    return run_retriever(state, retriever_agent)
    def answerer_node(state):     return run_answerer(state, answerer_agent)
    def critic_node(state):       return run_critic(state, critic_agent)
    def orchestrator_node(state): return run_orchestrator(state, orchestrator_agent)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve",    retriever_node)
    graph.add_node("answer",      answerer_node)
    graph.add_node("critique",    critic_node)
    graph.add_node("increment",   increment_revision)
    graph.add_node("orchestrate", orchestrator_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",  "answer")
    graph.add_edge("answer",    "critique")
    graph.add_edge("increment", "answer")
    graph.add_edge("orchestrate", END)

    graph.add_conditional_edges(
        "critique",
        route_after_critic,
        {"orchestrate": "orchestrate", "revise": "increment"},
    )
    return graph.compile()


# ─── Pipeline Entry Point ─────────────────────────────────────────────────────

class MultiAgentRAGPipeline:
    def __init__(
        self,
        collection_name: str = None,
        top_k: int = None,
    ):
        collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "rag_collection")
        top_k           = top_k or int(os.getenv("TOP_K", "5"))

        print("\n🤖  Initialising Multi-Agent RAG Pipeline (Fireworks AI)...")
        retriever    = Retriever(collection_name=collection_name, top_k=top_k)
        self.graph   = build_graph(
            RetrieverAgent(retriever=retriever),
            AnswererAgent(),
            CriticAgent(),
            OrchestratorAgent(),
        )
        print("✅  Pipeline ready\n")

    def run(self, query: str, verbose: bool = True) -> dict:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  QUERY: {query}")
            print(f"{'='*60}")

        state = self.graph.invoke({
            "query":            query,
            "retriever_result": None,
            "answerer_result":  None,
            "critic_result":    None,
            "final_result":     None,
            "revision_count":   0,
            "start_time":       time.perf_counter(),
            "node_timings":     {},
        })

        if verbose:
            self._print_summary(state)
        return state

    def _print_summary(self, state: RAGState):
        final   = state.get("final_result", {})
        timings = state.get("node_timings", {})
        critic  = state.get("critic_result", {})
        retr    = state.get("retriever_result", {})

        print(f"\n{'='*60}")
        print(f"  RESULT SUMMARY")
        print(f"{'='*60}")
        print(f"  Retrieval attempts : {retr.get('attempts', '?')}")
        print(f"  Retrieval score    : {retr.get('avg_score', 0):.2f}")
        print(f"  Critic verdict     : {critic.get('verdict', '?')}")
        print(f"  Revisions          : {state.get('revision_count', 0)}")
        print(f"  Confidence         : {final.get('confidence', 0):.2f}")
        print(f"  Total latency      : {timings.get('total', 0):.2f}s")
        flags = final.get("quality_flags", [])
        if flags:
            print(f"  ⚠️  Flags          : {flags}")
        print(f"\n  ANSWER:\n  {final.get('final_answer', 'No answer generated')}")
        print(f"{'='*60}\n")


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = MultiAgentRAGPipeline()
    # specific queries that map to exactly one paper
    queries = [
        "How does multi-head attention work in the Transformer?",
        "How does DQN use experience replay to train Atari games?",
        "What is BERT and how is it pre-trained?",
        "How does RAG combine retrieval with generation?",
        "What is the clipped surrogate objective in PPO?",
    ]
    for q in queries:
        pipeline.run(q)