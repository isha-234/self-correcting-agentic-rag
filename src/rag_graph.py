"""
rag_graph.py — LangGraph Multi-Agent Pipeline
===============================================
Wires the 3 agents into a directed state machine.

Flow:
  retrieve -> answer -> critique
                          approve -> finalize -> END
                          revise  -> answer (once) -> finalize -> END

Final assembly (merging the Critic's revision if any, and flagging
quality issues) is plain deterministic logic - no LLM call needed
since it's just branching on values already in state.
"""

import os
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from retriever import Retriever
from agents import RetrieverAgent, AnswererAgent, CriticAgent

from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

MAX_REVISIONS       = int(os.getenv("MAX_CRITIC_REVISIONS", "1"))
SCORE_THRESHOLD     = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.6"))
LOW_CONFIDENCE_FLAG = 0.5


# --- Shared State -------------------------------------------------------------

class RAGState(TypedDict):
    query:            str
    retriever_result: Optional[dict]
    answerer_result:  Optional[dict]
    critic_result:    Optional[dict]
    final_result:     Optional[dict]
    revision_count:   int
    start_time:       float
    node_timings:     dict


# --- Node Functions ------------------------------------------------------------

def run_retriever(state: RAGState, agent: RetrieverAgent) -> dict:
    t0     = time.perf_counter()
    print(f"\n[Graph] Running RetrieverAgent")
    result = agent.run(state["query"])
    timings = {**state.get("node_timings", {}), "retriever": round(time.perf_counter() - t0, 3)}
    return {"retriever_result": result, "node_timings": timings}


def run_answerer(state: RAGState, agent: AnswererAgent) -> dict:
    t0 = time.perf_counter()
    print(f"\n[Graph] Running AnswererAgent (revision #{state['revision_count']})")
    result  = agent.run(
        query=state["query"],
        context_string=state["retriever_result"]["context_string"],
    )
    key     = f"answerer_r{state['revision_count']}"
    timings = {**state.get("node_timings", {}), key: round(time.perf_counter() - t0, 3)}
    return {"answerer_result": result, "node_timings": timings}


def run_critic(state: RAGState, agent: CriticAgent) -> dict:
    t0 = time.perf_counter()
    print(f"\n[Graph] Running CriticAgent")
    result  = agent.run(
        query=state["query"],
        context_string=state["retriever_result"]["context_string"],
        draft_answer=state["answerer_result"].get("answer", ""),
    )
    timings = {**state.get("node_timings", {}), "critic": round(time.perf_counter() - t0, 3)}
    return {"critic_result": result, "node_timings": timings}


def finalize(state: RAGState) -> dict:
    """
    Plain deterministic aggregation - no LLM call.
    Uses the Critic's revised answer if it revised, otherwise the
    Answerer's original, and flags quality issues based on values
    already computed by the other agents.
    """
    t0 = time.perf_counter()
    print(f"\n[Graph] Running Finalize")

    retriever_result = state["retriever_result"]
    answerer_result  = state["answerer_result"]
    critic_result    = state["critic_result"]

    if critic_result.get("verdict") == "revise" and critic_result.get("revised_answer"):
        final_answer = critic_result["revised_answer"]
    else:
        final_answer = answerer_result.get("answer", "")

    quality_flags = []
    if retriever_result.get("avg_score", 0) < SCORE_THRESHOLD:
        quality_flags.append("low_retrieval_score")
    if answerer_result.get("confidence", 0) < LOW_CONFIDENCE_FLAG:
        quality_flags.append("low_confidence")
    if not answerer_result.get("answerable", True):
        quality_flags.append("unanswerable")

    result = {
        "final_answer": final_answer,
        "confidence":   answerer_result.get("confidence", 0),
        "quality_flags": quality_flags,
        "pipeline_summary": {
            "retrieval_attempts": retriever_result.get("attempts"),
            "retrieval_score":    retriever_result.get("avg_score"),
            "critic_verdict":     critic_result.get("verdict"),
            "critic_issues":      critic_result.get("issues", []),
        },
    }

    total   = time.perf_counter() - state["start_time"]
    timings = {
        **state.get("node_timings", {}),
        "finalize": round(time.perf_counter() - t0, 3),
        "total":    round(total, 3),
    }
    return {"final_result": result, "node_timings": timings}


def increment_revision(state: RAGState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


# --- Conditional Routing --------------------------------------------------------

def route_after_critic(state: RAGState) -> str:
    verdict        = state.get("critic_result", {}).get("verdict", "approve")
    revision_count = state.get("revision_count", 0)

    if verdict == "approve":
        print(f"\n[Graph] Critic approved, moving to Finalize")
        return "finalize"
    if revision_count >= MAX_REVISIONS:
        print(f"\n[Graph] Max revisions reached, moving to Finalize")
        return "finalize"

    print(f"\n[Graph] Critic rejected, moving to Answerer revision #{revision_count + 1}")
    return "revise"


# --- Graph Assembly --------------------------------------------------------------

def build_graph(
    retriever_agent: RetrieverAgent,
    answerer_agent:  AnswererAgent,
    critic_agent:    CriticAgent,
):
    def retriever_node(state): return run_retriever(state, retriever_agent)
    def answerer_node(state):  return run_answerer(state, answerer_agent)
    def critic_node(state):    return run_critic(state, critic_agent)
    def finalize_node(state):  return finalize(state)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve",  retriever_node)
    graph.add_node("answer",    answerer_node)
    graph.add_node("critique",  critic_node)
    graph.add_node("increment", increment_revision)
    graph.add_node("finalize",  finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",  "answer")
    graph.add_edge("answer",    "critique")
    graph.add_edge("increment", "answer")
    graph.add_edge("finalize",  END)

    graph.add_conditional_edges(
        "critique",
        route_after_critic,
        {"finalize": "finalize", "revise": "increment"},
    )
    return graph.compile()


# --- Pipeline Entry Point ----------------------------------------------------------

class MultiAgentRAGPipeline:
    def __init__(
        self,
        collection_name: str = None,
        top_k: int = None,
    ):
        collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "rag_collection")
        top_k           = top_k or int(os.getenv("TOP_K", "5"))

        print("\nInitialising Multi-Agent RAG Pipeline...")
        retriever  = Retriever(collection_name=collection_name, top_k=top_k)
        self.graph = build_graph(
            RetrieverAgent(retriever=retriever),
            AnswererAgent(),
            CriticAgent(),
        )
        print("Pipeline ready\n")

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
            print(f"  Flags              : {flags}")
        print(f"\n  ANSWER:\n  {final.get('final_answer', 'No answer generated')}")
        print(f"{'='*60}\n")


# --- Quick test ----------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = MultiAgentRAGPipeline()
    queries = [
        "How does multi-head attention work in the Transformer?",
        "How does DQN use experience replay to train Atari games?",
        "What is BERT and how is it pre-trained?",
        "How does RAG combine retrieval with generation?",
        "What is the clipped surrogate objective in PPO?",
    ]
    for q in queries:
        pipeline.run(q)