"""
agents.py — The 4 Independent Agent Definitions
=================================================
Every agent uses the Fireworks AI API via the OpenAI-compatible client.
Model names and API credentials are read from .env — no hardcoded values.

Agent roster:
  1. RetrieverAgent   — grades chunks, rewrites queries (llama-8b, cheap)
  2. AnswererAgent    — generates cited answers        (llama-70b, quality)
  3. CriticAgent      — fact-checks the answer         (llama-8b, cheap)
  4. OrchestratorAgent — synthesizes final output      (llama-70b, quality)
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

# ─── Fireworks client (OpenAI-compatible) ─────────────────────────────────────
# Setting base_url redirects all SDK calls to Fireworks.
# The variable is still called OPENAI_API_KEY because that's what
# the SDK reads by default — it just holds your Fireworks key.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
)

# ─── Model names from .env ────────────────────────────────────────────────────
MODEL_RETRIEVER    = os.getenv("MODEL_RETRIEVER",    "accounts/fireworks/models/llama-v3p1-8b-instruct")
MODEL_ANSWERER     = os.getenv("MODEL_ANSWERER",     "accounts/fireworks/models/llama-v3p1-70b-instruct")
MODEL_CRITIC       = os.getenv("MODEL_CRITIC",       "accounts/fireworks/models/llama-v3p1-8b-instruct")
MODEL_ORCHESTRATOR = os.getenv("MODEL_ORCHESTRATOR", "accounts/fireworks/models/llama-v3p1-70b-instruct")

SCORE_THRESHOLD  = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.6"))
MAX_ATTEMPTS     = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS",       "2"))
MAX_REVISIONS    = int(os.getenv("MAX_CRITIC_REVISIONS",         "1"))


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, model: str, system_prompt: str, temperature: float = 0.2):
        self.model         = model
        self.system_prompt = system_prompt
        self.temperature   = temperature

    def _call(self, user_message: str) -> str:
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_json(self, user_message: str) -> dict:
        """
        JSON mode — asks the model to return valid JSON only.
        Note: Fireworks supports response_format for most models,
        but if you hit a parsing error it falls back to a retry.
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            # Fallback: retry without json_object mode, ask explicitly for JSON
            raw = self._call(
                user_message + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no backticks."
            )
            # Strip any ```json fences if the model adds them
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)


# ─── Agent 1: RetrieverAgent ──────────────────────────────────────────────────

class RetrieverAgent(BaseAgent):
    """
    Uses llama-8b (fast, cheap) — runs on every query, possibly twice.
    Grades retrieved chunks and rewrites the query if quality is low.
    """

    SYSTEM = """You are a Retrieval Quality Agent. Your only job is to assess
whether retrieved document chunks genuinely answer a given question.

Score each chunk:
  1.0 — directly and completely answers the question
  0.7 — partially relevant or contains useful facts
  0.4 — tangentially related
  0.0 — irrelevant

If avg score < 0.6, provide a rewritten query using different vocabulary.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "scores": [{"chunk_index": 0, "score": 0.85, "reason": "..."}],
  "avg_score": 0.85,
  "quality": "high",
  "rewritten_query": null
}
quality must be "high" if avg_score >= 0.6, else "low"."""

    def __init__(self, retriever):
        super().__init__(model=MODEL_RETRIEVER, system_prompt=self.SYSTEM, temperature=0.1)
        self.retriever = retriever

    def run(self, query: str) -> dict:
        current_query = query

        for attempt in range(1, MAX_ATTEMPTS + 1):
            context_string, chunks = self.retriever.retrieve_as_context_string(
                current_query, top_k=int(os.getenv("TOP_K", "5"))
            )

            if not chunks:
                return {
                    "chunks": [], "context_string": "",
                    "rewritten_query": None, "attempts": attempt,
                    "avg_score": 0.0, "quality": "low",
                }

            result = self._call_json(
                f"Question: {query}\n\nRetrieved chunks:\n{context_string}\n\n"
                f"Grade each chunk's relevance to the question."
            )

            avg_score = result.get("avg_score", 0.0)
            quality   = result.get("quality", "low")

            best = {
                "chunks": chunks, "context_string": context_string,
                "rewritten_query": result.get("rewritten_query"),
                "attempts": attempt, "avg_score": avg_score,
                "quality": quality, "scores": result.get("scores", []),
            }

            if quality == "high" or avg_score >= SCORE_THRESHOLD:
                print(f"  [RetrieverAgent] ✅ score={avg_score:.2f} attempt={attempt}")
                return best

            rewritten = result.get("rewritten_query")
            if rewritten and attempt < MAX_ATTEMPTS:
                print(f"  [RetrieverAgent] ↩️  score={avg_score:.2f} — rewriting query")
                print(f"     Original : \"{current_query}\"")
                print(f"     Rewritten: \"{rewritten}\"")
                current_query = rewritten
            else:
                print(f"  [RetrieverAgent] ⚠️  max attempts — using best context (score={avg_score:.2f})")
                return best

        return best


# ─── Agent 2: AnswererAgent ───────────────────────────────────────────────────

class AnswererAgent(BaseAgent):
    """
    Uses llama-70b (quality) — this is the user-facing answer.
    Generates cited answer using ONLY the approved chunks.
    """

    SYSTEM = """You are a precise Answer Generation Agent.

Rules:
  1. Use ONLY information from the provided chunks — no outside knowledge
  2. Cite every factual claim inline: [1], [2] etc.
  3. If chunks lack sufficient info, say so explicitly
  4. Be concise — one paragraph unless complexity requires more

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "answer": "The transformer uses [1] attention...",
  "citations": [{"id": 1, "source": "paper.pdf", "page": 3, "quote": "short quote < 20 words"}],
  "confidence": 0.87,
  "answerable": true
}"""

    def __init__(self):
        super().__init__(model=MODEL_ANSWERER, system_prompt=self.SYSTEM, temperature=0.2)

    def run(self, query: str, context_string: str) -> dict:
        print(f"  [AnswererAgent] 📝 Generating answer...")
        result = self._call_json(
            f"Question: {query}\n\n"
            f"Document chunks (use ONLY these):\n{context_string}\n\n"
            f"Generate a cited answer."
        )
        print(f"  [AnswererAgent] ✅ confidence={result.get('confidence', 0):.2f}")
        return result


# ─── Agent 3: CriticAgent ─────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """
    Uses llama-8b (fast) — verification task, not generation.
    Independently fact-checks the AnswererAgent's draft.
    Adversarial by design — no loyalty to the Answerer.
    """

    SYSTEM = """You are a rigorous Fact-Checking Agent. You review answers
generated by another AI. Your job is adversarial — assume errors may exist.

Check for:
  1. Hallucinations — claims not in any source chunk
  2. Missing citations — facts stated without [N] reference
  3. Misattribution — claim cited to wrong chunk
  4. Overconfidence — strong claims where sources are ambiguous

Verdict rules:
  "approve" — answer is substantially accurate and grounded in the chunks.
              Minor rephrasing of source text is acceptable.
              Mathematical notation may differ in form from the source.
              Approve if the core claims are supported, even if wording varies.
  "revise"  — only if there is a clear factual hallucination (a claim that
              directly contradicts the source), or a citation pointing to a
              completely wrong chunk. Do NOT revise for style or minor phrasing.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "verdict": "approve",
  "issues": [],
  "revised_answer": null,
  "critique_confidence": 0.92
}"""

    def __init__(self):
        super().__init__(model=MODEL_CRITIC, system_prompt=self.SYSTEM, temperature=0.1)

    def run(self, query: str, context_string: str, draft_answer: str) -> dict:
        print(f"  [CriticAgent] 🔍 Reviewing draft answer...")
        result = self._call_json(
            f"Original question: {query}\n\n"
            f"Source chunks (ground truth):\n{context_string}\n\n"
            f"Draft answer to review:\n{draft_answer}\n\n"
            f"Find any hallucinations, missing citations, or misattributions."
        )
        verdict = result.get("verdict", "revise")
        issues  = result.get("issues", [])
        if verdict == "approve":
            print(f"  [CriticAgent] ✅ Approved")
        else:
            print(f"  [CriticAgent] ❌ Revision needed — {len(issues)} issue(s)")
            for issue in issues:
                print(f"     • {issue}")
        return result


# ─── Agent 4: OrchestratorAgent ───────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    Uses llama-70b (quality) — needs full reasoning for edge cases.
    The ONLY agent aware of all other agents.
    Synthesizes final output with quality flags and pipeline metadata.
    """

    SYSTEM = """You are an Orchestration Agent managing a multi-agent RAG pipeline.
You receive outputs from three specialist agents and produce the final response.

Your job:
  1. Use the Critic's revised answer if verdict was "revise", otherwise use the original
  2. Add quality_flags for: low retrieval score (<0.6), low confidence (<0.5), unanswerable
  3. Produce a clean final response with pipeline metadata

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "final_answer": "...",
  "confidence": 0.85,
  "quality_flags": [],
  "pipeline_summary": {
    "retrieval_attempts": 1,
    "retrieval_score": 0.81,
    "critic_verdict": "approve",
    "critic_issues": []
  }
}"""

    def __init__(self):
        super().__init__(model=MODEL_ORCHESTRATOR, system_prompt=self.SYSTEM, temperature=0.1)

    def run(
        self,
        query: str,
        retriever_result: dict,
        answerer_result:  dict,
        critic_result:    dict,
    ) -> dict:
        print(f"  [OrchestratorAgent] 🎯 Synthesizing final answer...")
        result = self._call_json(
            f"Query: {query}\n\n"
            f"Retriever output:\n"
            f"  attempts={retriever_result['attempts']}, "
            f"  avg_score={retriever_result['avg_score']:.2f}, "
            f"  quality={retriever_result['quality']}\n\n"
            f"Answerer output:\n"
            f"  answer={answerer_result.get('answer', 'N/A')}\n"
            f"  confidence={answerer_result.get('confidence', 0):.2f}\n"
            f"  answerable={answerer_result.get('answerable', True)}\n\n"
            f"Critic output:\n"
            f"  verdict={critic_result.get('verdict', 'revise')}\n"
            f"  issues={critic_result.get('issues', [])}\n"
            f"  revised_answer={critic_result.get('revised_answer', 'None')}\n\n"
            f"Synthesize the final response."
        )
        print(f"  [OrchestratorAgent] ✅ confidence={result.get('confidence', 0):.2f}")
        return result
