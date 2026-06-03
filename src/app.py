"""
app.py — Streamlit Demo UI
===========================
Run:
    streamlit run app.py
    (api.py must be running on port 8000 first)
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Self-Correcting RAG")
st.caption("4 independent agents: Retriever → Answerer → Critic → Orchestrator")

with st.sidebar:
    st.header("Config")
    top_k   = st.slider("Top-K chunks", 3, 10, 5)
    verbose = st.checkbox("Verbose logging", value=False)
    st.divider()
    st.markdown("**Agents**")
    st.markdown("RetrieverAgent — llama-8b")
    st.markdown("AnswererAgent — llama-70b")
    st.markdown("CriticAgent — llama-8b")
    st.markdown("OrchestratorAgent — llama-70b")
    st.divider()

    # Health check
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API: {h['status']}")
    except Exception:
        st.error("API offline — run: uvicorn api:app --reload")

query   = st.text_input("Ask a question", placeholder="What is the main contribution of this work?")
run_btn = st.button("Run Pipeline", type="primary", disabled=not query.strip())

if run_btn and query.strip():
    with st.spinner("Running 4 agents..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"query": query, "top_k": top_k, "verbose": verbose},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("API not running. Start with: `uvicorn api:app --reload --port 8000`")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.divider()
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Answer")
        st.markdown(data["final_answer"])
        for flag in data.get("quality_flags", []):
            st.warning(f"⚠️ {flag}")

    with col2:
        st.metric("Confidence",       f"{data['confidence']:.0%}")
        st.metric("Retrieval Score",  f"{data['retrieval_score']:.2f}")
        st.metric("Latency",          f"{data['latency_seconds']:.1f}s")

    st.divider()
    st.subheader("Agent Trace")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        score = data["retrieval_score"]
        st.markdown("**🔵 Retriever**")
        st.markdown(f"{'🟢' if score >= 0.6 else '🟡'} Score: `{score:.2f}`")
        st.markdown(f"Attempts: `{data['retrieval_attempts']}`")
        st.caption(f"⏱ {data['node_timings'].get('retriever', 0):.2f}s")

    with c2:
        st.markdown("**🟢 Answerer**")
        st.markdown(f"Revisions: `{data['revision_count']}`")
        st.caption(f"⏱ {data['node_timings'].get('answerer_r0', 0):.2f}s")

    with c3:
        verdict = data["critic_verdict"]
        st.markdown("**🔴 Critic**")
        st.markdown(f"{'✅' if verdict == 'approve' else '❌'} `{verdict}`")
        st.caption(f"⏱ {data['node_timings'].get('critic', 0):.2f}s")

    with c4:
        st.markdown("**🟡 Orchestrator**")
        st.markdown("Final synthesis")
        st.caption(f"⏱ {data['node_timings'].get('orchestrator', 0):.2f}s")

    if data["revision_count"] > 0:
        st.info(f"🔄 Critic requested {data['revision_count']} revision(s) — final answer is the revised version.")
