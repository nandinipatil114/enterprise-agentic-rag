"""Streamlit dashboard: chat, live graph execution log, and evaluation metrics."""
import json
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import get_graph, stream_agent
from config import COLLECTION, EMBEDDING_MODEL, LLM_MODEL, QDRANT_PATH, get_qdrant
from evaluator import RESULTS_PATH, run_evaluation

st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="🧭", layout="wide")

NODE_LABEL = {
    "agent": "🧠 agent", "retrieve": "🔎 retrieve", "grade_documents": "⚖️ grade_documents",
    "rewrite_query": "✍️ rewrite_query", "calculate": "🧮 calculate", "generate": "💬 generate",
}
EXAMPLES = [
    "How many leave days can I carry over?",
    "What's the hotel limit in London?",
    "What credit do I get if uptime is 97%?",
    "What is 24 days * 8 hours?",
]


def kb_status() -> tuple[bool, str | None]:
    """Return (ready, error_message)."""
    try:
        client = get_qdrant()
        return client.collection_exists(COLLECTION) and client.count(COLLECTION).count > 0, None
    except Exception as exc:
        return False, str(exc)


def describe(node: str, upd: dict | None, ctx: dict) -> str:
    upd = upd or {}
    if node == "agent":
        msg = upd["messages"][0]
        if msg.tool_calls:
            call = msg.tool_calls[0]
            return f"Router chose tool `{call['name']}` with args `{json.dumps(call['args'])}`"
        return "Router answered directly (no tool needed)"
    if node == "retrieve":
        ctx["retrieved"] = len(upd.get("documents", []))
        return f"Retrieved {ctx['retrieved']} chunks from Qdrant"
    if node == "grade_documents":
        docs = upd.get("documents", [])
        ctx["sources"] = [f"[{i}] {d['source']} · p.{d['page']}" for i, d in enumerate(docs, 1)]
        return f"Grader kept {len(docs)}/{ctx.get('retrieved', 0)} chunks as relevant"
    if node == "rewrite_query":
        return f"Chunks irrelevant, query rewritten → “{upd['search_query']}” (attempt {upd['rewrite_count']})"
    if node == "calculate":
        return f"Calculator returned `{upd['messages'][0].content}`"
    return f"Answer generated from {len(ctx.get('sources', []))} graded sources"


def to_lc(history: list[dict]):
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history[-6:]]


def render_trace(trace: list[str], sources: list[str]) -> None:
    with st.expander("Execution trace & sources"):
        for line in trace:
            st.markdown(line)
        if sources:
            st.markdown("**Sources**")
            for s in sources:
                st.caption(s)


# ------------------------------------------------------------------ sidebar
ready, kb_error = kb_status()

with st.sidebar:
    st.title("🧭 Agentic RAG")
    st.caption(f"LLM: `{LLM_MODEL}` (Groq)  \nEmbeddings: `{EMBEDDING_MODEL}`  \nVector DB: Qdrant (`{COLLECTION}`)")
    if ready:
        st.success("Knowledge base ready")
    elif kb_error:
        st.error(f"Could not open the vector store: {kb_error}")
    else:
        st.error("Knowledge base is empty. Stop this app (Ctrl+C), run `python ingest.py`, then start it again.")
        st.caption(f"Looking in: {QDRANT_PATH}")
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch", disabled=not ready):
            st.session_state["pending"] = ex
    if st.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
    with st.expander("Graph definition (Mermaid)"):
        try:
            st.code(get_graph().get_graph().draw_mermaid(), language="text")
        except Exception as exc:
            st.caption(f"Graph diagram unavailable: {exc}")

if not ready:
    st.stop()

st.session_state.setdefault("messages", [])
chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Evaluation"])
prompt = st.chat_input("Ask about HR, IT security, expenses, or the cloud SLA…") or st.session_state.pop("pending", None)

# ------------------------------------------------------------------ chat tab
with chat_tab:
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                render_trace(m["trace"], m.get("sources", []))

    if prompt:
        history = to_lc(st.session_state["messages"])
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            trace, ctx, answer = [], {}, ""
            try:
                with st.status("Running agent graph…", expanded=True) as status:
                    for step in stream_agent(prompt, history):
                        for node, upd in step.items():
                            line = f"**{NODE_LABEL.get(node, node)}** — {describe(node, upd, ctx)}"
                            st.markdown(line)
                            trace.append(line)
                            answer = (upd or {}).get("answer", answer)
                    status.update(label="Graph finished", state="complete", expanded=False)
                st.markdown(answer)
                sources = ctx.get("sources", [])
                for s in sources:
                    st.caption(s)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "trace": trace, "sources": sources}
                )
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")

# ------------------------------------------------------------------ evaluation tab
with eval_tab:
    st.markdown("Scores the agent on a golden Q&A set: **Faithfulness**, **Answer Relevancy**, **Context Recall**.")
    backend = st.radio(
        "Backend", ["custom", "ragas"], horizontal=True,
        help="'custom' is an LLM judge that works reliably on Groq; 'ragas' uses the Ragas library and may hit rate limits.",
    )
    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        try:
            run_evaluation(backend, on_progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    if Path(RESULTS_PATH).exists():
        res = json.loads(Path(RESULTS_PATH).read_text())
        st.caption(f"Backend: {res['backend']} · Model: {res['model']} · Run at {res['timestamp']}")
        cols = st.columns(3)
        for col, (key, label) in zip(
            cols, [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy"), ("context_recall", "Context Recall")]
        ):
            val = res["summary"].get(key)
            col.metric(label, "n/a" if val is None else f"{val:.2f}")
        st.dataframe(
            [
                {
                    "question": r["question"], "faithfulness": r.get("faithfulness"),
                    "answer_relevancy": r.get("answer_relevancy"), "context_recall": r.get("context_recall"),
                    "rewrites": r["rewrites"],
                }
                for r in res["rows"]
            ]
        )
        with st.expander("Answers vs. references"):
            for r in res["rows"]:
                st.markdown(f"**Q:** {r['question']}\n\n**Agent:** {r['response']}\n\n**Reference:** {r['reference']}")
                st.divider()
    else:
        st.info("No evaluation results yet. Click **Run evaluation**.")